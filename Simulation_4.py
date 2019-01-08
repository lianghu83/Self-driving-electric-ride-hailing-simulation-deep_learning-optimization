"""
Run taxi system simulation.
"""

import pandas as pd
import numpy as np
from gurobipy import tuplelist
from geopy.distance import great_circle
import math
import itertools
from copy import deepcopy
from datetime import datetime
from Parameters import *
from Dispatcher import *
#from MyFunctions import *
#from Classes_2 import *
import random
import os
import shutil
from matplotlib import pyplot as plt
import math

#%%

#setup scenario
#if os.path.exists('H:\\EAV Taxi Data\\'+scenario):
#    shutil.rmtree('H:\\EAV Taxi Data\\'+scenario)
os.mkdir('H:\\EAV Taxi Data\\'+scenario)
os.mkdir('H:\\EAV Taxi Data\\'+scenario+'\\Dispatch')

#%%
#record running time
optimization_times = []
optimization_dispatch_times = []
deep_learning_times = []

#start time
start_running = datetime.now() 
print(start_running)

#%%initiate big request status table
#select dataset
data_used = "10_16_requests_5%"
#read trip data
df = pd.read_csv('H:\\EAV Taxi Data\\Raw Data\\'+data_used+'.csv')
df.drop(['hack_license','vendor_id','rate_code','store_and_fwd_flag','passenger_count'], axis='columns', inplace=True)
df.columns
#change charging power to 50kW
df['CS_power'] = charging_power
#distance to charging station
df['CS_travel_time_in_mins'].describe()

Request = df.copy(deep=True)
Request['id'] = df.index
Request.index = Request['id']
#adjust Request table
Request.drop(['medallion', 'trip_time_in_secs', 'CS_power'], axis='columns', inplace=True)    
Request = Request[['id','pickup_datetime','pickup_longitude','pickup_latitude',\
                   'trip_distance','trip_time_in_mins','dropoff_datetime','dropoff_longitude',\
                   'dropoff_latitude','CS_distance','CS_travel_time_in_mins','CS_longitude','CS_latitude','date']]
Request.columns = ['id','origin_timestamp','origin_longitude','origin_latitude',\
                   'trip_distance','trip_time','destination_timestamp','destination_longitude',\
                   'destination_latitude','cs_distance','cs_travel_time','cs_longitude','cs_latitude','date']
Request['wait_time'] = 0
Request['served'] = False
Request['served_taxi'] = -1
Request['taxi_pickup_time'] = -1
Request['pickup_timestamp'] = -1
Request['dropoff_timestamp'] = -1
Request_col = Request.columns.tolist()
Request['cs_distance'].describe()

#request get rejected at a time interval
def getRejected(request_id):
    Request['wait_time'][request_id] += 1
    
#request get accepted, a taxi comes
def getAccepted(request_id, medallion, taxi_pickup_time):
    Request['wait_time'][request_id] += taxi_pickup_time
    Request['served'][request_id] = True
    Request['served_taxi'][request_id] = medallion
    Request['taxi_pickup_time'][request_id] = taxi_pickup_time
    
#request is served by taxi    
def getServed(request_id, start_timestamp):
    Request['pickup_timestamp'][request_id] = start_timestamp
    Request['dropoff_timestamp'][request_id] = start_timestamp + Request['trip_time'][request_id]*60

#%%initiate big taxi status table
data_used = "10_16_taxis_5%"
df = pd.read_csv('H:\\EAV Taxi Data\\Raw Data\\'+data_used+'.csv') 
#drop 2013000000
df['medallion'] = df['medallion'] % 2013000000
    
medallion = sorted(list(set(df['medallion'])))
#cut fleet size to test
fleet_size = math.ceil(len(medallion)*0.95)
np.random.seed(fleet_size) #seed=number of taxis
medallion = sorted(list(np.random.choice(medallion, fleet_size, replace=False)))

#create big taxi status table
Taxi = pd.DataFrame(columns=['medallion','status','start_timestamp','start_longitude',\
                             'start_latitude','start_range','status_distance','status_time',\
                             'end_timestamp','end_longitude','end_latitude','end_range'])
#initiate the taxi status table
random.seed(1991)    
for i in range(len(medallion)):
    each_taxi = df[df['medallion']==medallion[i]]
    each_taxi = each_taxi.sort_values(['pickup_datetime'])
    each_taxi = each_taxi.reset_index(drop=True)
    Taxi.loc[i, 'medallion'] = medallion[i]
    Taxi.loc[i, 'status'] = 'waiting'
    Taxi.loc[i, 'start_timestamp'] = start_time
    Taxi.loc[i, 'start_longitude'] = each_taxi.loc[0, 'pickup_longitude']
    Taxi.loc[i, 'start_latitude'] = each_taxi.loc[0, 'pickup_latitude']
    temp_range = random.uniform(low_range, EV_range)
    Taxi.loc[i, 'start_range'] = temp_range
    Taxi.loc[i, 'status_distance'] = 0.0
    Taxi.loc[i, 'status_time'] = -1
    Taxi.loc[i, 'end_timestamp'] = -1
    Taxi.loc[i, 'end_longitude'] = each_taxi.loc[0, 'pickup_longitude']
    Taxi.loc[i, 'end_latitude'] = each_taxi.loc[0, 'pickup_latitude']
    Taxi.loc[i, 'end_range'] = temp_range
Taxi.index = medallion
Taxi_col = Taxi.columns.tolist()
Taxi['start_range'].describe()
Taxi.dtypes

#create a taxi activity table to store all activities
Taxi_activities = Taxi.copy(deep=True)
Taxi_activities = Taxi_activities.reset_index(drop=True)

#taxi wait at somewhere
def startWaiting(medallion, timestamp):
    which_taxi = Taxi.loc[medallion]
    which_taxi['status'] = 'waiting'
    which_taxi['start_timestamp'] = timestamp
    which_taxi['start_longitude'] = which_taxi['end_longitude']
    which_taxi['start_latitude'] = which_taxi['end_latitude']
    which_taxi['start_range'] = which_taxi['end_range']
    which_taxi['status_distance'] = 0.0
    which_taxi['status_time'] = -1
    which_taxi['end_timestamp'] = -1
    which_taxi['end_longitude'] = which_taxi['start_longitude']
    which_taxi['end_latitude'] = which_taxi['start_latitude']
    which_taxi['end_range'] = which_taxi['start_range']
    Taxi.update(which_taxi)
    global Taxi_activities
    Taxi_activities = Taxi_activities.append(which_taxi, ignore_index=True)

#taxi gets called to pickup customer
def getCalled(medallion, request_id, timestamp, taxi_pickup_dist, taxi_pickup_time):
    which_taxi = Taxi.loc[medallion]
    which_request = Request.loc[request_id][['origin_longitude', 'origin_latitude']]
    which_taxi['status'] = 'called'
    which_taxi['start_timestamp'] = timestamp
    which_taxi['start_longitude'] = which_taxi['end_longitude']
    which_taxi['start_latitude'] = which_taxi['end_latitude']
    which_taxi['start_range'] = which_taxi['end_range']
    which_taxi['status_distance'] = taxi_pickup_dist
    which_taxi['status_time'] = taxi_pickup_time
    which_taxi['end_timestamp'] = which_taxi['start_timestamp'] + taxi_pickup_time*60
    which_taxi['end_longitude'] = which_request['origin_longitude']
    which_taxi['end_latitude'] = which_request['origin_latitude']
    which_taxi['end_range'] = which_taxi['start_range'] - taxi_pickup_dist
    Taxi.update(which_taxi)
    global Taxi_activities
    Taxi_activities = Taxi_activities.append(which_taxi, ignore_index=True)

#taxi serves customer
def serveCustomer(medallion, request_id, timestamp):
    which_taxi = Taxi.loc[medallion]
    which_request = Request.loc[request_id][['trip_distance', 'trip_time', 'destination_longitude', 'destination_latitude']]
    which_taxi['status'] = 'occupied'
    which_taxi['start_timestamp'] = timestamp
    which_taxi['start_longitude'] = which_taxi['end_longitude']
    which_taxi['start_latitude'] = which_taxi['end_latitude']
    which_taxi['start_range'] = which_taxi['end_range']
    which_taxi['status_distance'] = which_request['trip_distance']
    which_taxi['status_time'] = which_request['trip_time']
    which_taxi['end_timestamp'] = which_taxi['start_timestamp'] + which_taxi['status_time']*60
    which_taxi['end_longitude'] = which_request['destination_longitude']
    which_taxi['end_latitude'] = which_request['destination_latitude']
    which_taxi['end_range'] = which_taxi['start_range'] - which_taxi['status_distance']
    Taxi.update(which_taxi)
    global Taxi_activities
    Taxi_activities = Taxi_activities.append(which_taxi, ignore_index=True)

#taxi goes to charging station
def goToCharging(medallion, request_id, timestamp):
    which_taxi = Taxi.loc[medallion]
    which_request = Request.loc[request_id][['cs_distance', 'cs_travel_time', 'cs_longitude', 'cs_latitude']]
    which_taxi['status'] = 'go charging'
    which_taxi['start_timestamp'] = timestamp
    which_taxi['start_longitude'] = which_taxi['end_longitude']
    which_taxi['start_latitude'] = which_taxi['end_latitude']
    which_taxi['start_range'] = which_taxi['end_range']
    which_taxi['status_distance'] =  which_request['cs_distance']
    which_taxi['status_time'] =  which_request['cs_travel_time']
    which_taxi['end_timestamp'] = which_taxi['start_timestamp'] + which_taxi['status_time']*60
    which_taxi['end_longitude'] =  which_request['cs_longitude']
    which_taxi['end_latitude'] =  which_request['cs_latitude']
    which_taxi['end_range'] = which_taxi['start_range'] - which_taxi['status_distance']
    Taxi.update(which_taxi)
    global Taxi_activities
    Taxi_activities = Taxi_activities.append(which_taxi, ignore_index=True)

#taxi starts charging
def startCharging(medallion, timestamp):
    which_taxi = Taxi.loc[medallion]
    which_taxi['status'] = 'start charging'
    which_taxi['start_timestamp'] = timestamp
    which_taxi['start_longitude'] = which_taxi['end_longitude']
    which_taxi['start_latitude'] = which_taxi['end_latitude']
    which_taxi['start_range'] = which_taxi['end_range']
    which_taxi['status_distance'] = 0.0
    which_taxi['status_time'] = -1
    which_taxi['end_timestamp'] = -1 #self.start_timestamp + self.status_time*60
    which_taxi['end_longitude'] = which_taxi['start_longitude']
    which_taxi['end_latitude'] = which_taxi['start_latitude']
    which_taxi['end_range'] = which_taxi['start_range']
    Taxi.update(which_taxi)
    global Taxi_activities
    Taxi_activities = Taxi_activities.append(which_taxi, ignore_index=True)

#taxi ends charging
def endCharging(medallion, timestamp):
    which_taxi = Taxi.loc[medallion]
    which_taxi['status'] = 'waiting'
    add_range = min((timestamp-which_taxi['start_timestamp'])/3600*charging_power/electricity_consumption_rate, EV_range-which_taxi['start_range'])
    which_taxi['start_timestamp'] = timestamp
    which_taxi['start_longitude'] = which_taxi['end_longitude']
    which_taxi['start_latitude'] = which_taxi['end_latitude']
    which_taxi['start_range'] = which_taxi['start_range'] + add_range
    which_taxi['status_distance'] = 0.0
    which_taxi['status_time'] = -1
    which_taxi['end_timestamp'] = -1
    which_taxi['end_longitude'] = which_taxi['start_longitude']
    which_taxi['end_latitude'] = which_taxi['start_latitude']
    which_taxi['end_range'] = which_taxi['start_range']
    Taxi.update(which_taxi)
    global Taxi_activities
    Taxi_activities = Taxi_activities.append(which_taxi, ignore_index=True)

#%%iterate over every 1 minute
for j in range(len(time_range)):
    
    if (j%20==0):
        print(j)
        
    #update tables first
    #update taxi SOC in the Taxi table
    for t in Taxi.index:
        if ((Taxi['status'][t]=='start charging') & (Taxi['start_timestamp'][t]<time_range[j])):
            Taxi['end_range'][t] += time_interval/3600*charging_power/electricity_consumption_rate
            if Taxi['end_range'][t]>=EV_range:
                Taxi['end_range'][t] = EV_range
                endCharging(t, time_range[j])
    
    ### Preparation for dispatch ###
    
    #extract taxis and requests within this time interval
    taxi_sub = Taxi[(Taxi['end_timestamp']<(time_range[j]+time_interval)) &
                    ((Taxi['status']=='waiting') | ((Taxi['status']=='start charging') & (Taxi['end_range']>=charge_to))) &
                    #(Taxi['status']=='waiting') &
                    (Taxi['start_timestamp']<(time_range[j]+time_interval))]
    request_sub = Request[(Request['served']==False) & 
                          (Request['wait_time']<=max_wait_time) &
                          (Request['origin_timestamp']<(time_range[j]+time_interval))]

    #get index, as keys for dictionary
    taxi_sub_index = taxi_sub.index.tolist()
    request_sub_index = request_sub.index.tolist()
    match_path = tuplelist(list(itertools.product(taxi_sub_index, request_sub_index)))

    #if no taxi and no request
    if ((len(taxi_sub_index)==0) & (len(request_sub_index)==0)):
        continue
    #if no request only
    if ((len(taxi_sub_index)>0) & (len(request_sub_index)==0)):
        continue
    #if no taxi only
    if ((len(taxi_sub_index)==0) & (len(request_sub_index)>0)):
        for each_request in request_sub_index:         
            getRejected(each_request)
        continue
    
    #convert dataframe into dictionary
    taxi_sub = taxi_sub.to_dict(orient="index")
    request_sub = request_sub.to_dict(orient="index")

    #calculate pickup distance dictionary
    pickup_distance = pd.DataFrame(index=taxi_sub_index, columns=request_sub_index)
    pickup_distance = pickup_distance.to_dict(orient="index")
    for each_taxi in taxi_sub_index:
        for each_request in request_sub_index:
            taxi_GPS = (taxi_sub[each_taxi]['end_latitude'], taxi_sub[each_taxi]['end_longitude']) #latitude first
            request_GPS = (request_sub[each_request]['origin_latitude'], request_sub[each_request]['origin_longitude'])
            pickup_distance[each_taxi][each_request] = 1.4413*great_circle(taxi_GPS, request_GPS).miles + 0.1383 #miles

    #calculate pickup time dictionary
    pickup_time_in_mins = deepcopy(pickup_distance)
    for each_taxi in taxi_sub_index:
        for each_request in request_sub_index:
            pickup_time_in_mins[each_taxi][each_request] = int(math.ceil(
                    pickup_time_in_mins[each_taxi][each_request]/speed_average_NYC*60))
             
    ### Apply dispatch rules ###
    """
    #CENTRALIZED OPTIMIZATION
    start_running_1 = datetime.now()
    v = CentralizedOptimization3(taxi_sub,
                                 taxi_sub_index,
                                 request_sub,
                                 request_sub_index,
                                 match_path,
                                 pickup_distance,
                                 pickup_time_in_mins)
    optimization_times.append((datetime.now()-start_running_1).total_seconds())
    
    #get dispatch data
    dispatch_data = CombineTaxiRequest(taxi_sub,
                                       Taxi_col,
                                       request_sub,
                                       Request_col,
                                       pickup_distance,
                                       pickup_time_in_mins,
                                       v,
                                       time_range[j])
    optimization_dispatch_times.append((datetime.now()-start_running_1).total_seconds())
    dispatch_data.to_csv('H:\\EAV Taxi Data\\'+scenario+'\\'+'Dispatch\\'+str(j)+'.csv', index=False)
    """
    #MACHINE LEARNING
    X, df_temp = DeepLearningDispatchPrepareData(taxi_sub,
                                                 Taxi_col,
                                                 request_sub,
                                                 Request_col,                       
                                                 pickup_distance,
                                                 pickup_time_in_mins,
                                                 time_range[j])
    start_running_1 = datetime.now()
    v = DeepLearningDispatch2(X, df_temp, taxi_sub_index, request_sub_index)
    deep_learning_times.append((datetime.now()-start_running_1).total_seconds())
    
    ### Update taxi status and request status ###
    
    #(1) when taxi-request match
    for each_taxi in taxi_sub_index:
        for each_request in request_sub_index:
            if (v[each_taxi][each_request]==1):
                #excute dispatch actions
                if Taxi['status'][each_taxi]=='start charging':
                    endCharging(each_taxi, time_range[j])
                pickup_time_taxi_request = pickup_time_in_mins[each_taxi][each_request]
                getCalled(each_taxi, each_request, time_range[j],\
                          pickup_distance[each_taxi][each_request],\
                          pickup_time_taxi_request)
                getAccepted(each_request, each_taxi, pickup_time_taxi_request)
                serveCustomer(each_taxi, each_request, Taxi['end_timestamp'][each_taxi])
                getServed(each_request, Taxi['start_timestamp'][each_taxi])
                #check charging
                if (Request['cs_distance'][each_request] <= Taxi['end_range'][each_taxi] <= low_range):
                    goToCharging(each_taxi, each_request, Taxi['end_timestamp'][each_taxi])
                    startCharging(each_taxi, Taxi['end_timestamp'][each_taxi])
                else:
                    startWaiting(each_taxi, Taxi['end_timestamp'][each_taxi])

    """                
    #(2) when taxi not called
    for each_taxi in taxi_sub_index:
        if (sum(v[each_taxi].values())==0):
            pass #do nothing
    """
    
    #(3) when request not accepted
    for each_request in request_sub_index:
        v_sum = 0
        for each_taxi in taxi_sub_index:
            v_sum += v[each_taxi][each_request]  
        if (v_sum==0):
            getRejected(each_request)



#end time
print(datetime.now()-start_running)

#%%write output

Taxi.to_csv('H:\\EAV Taxi Data\\'+scenario+'\\taxi_output.csv', index=False)
Request.to_csv('H:\\EAV Taxi Data\\'+scenario+'\\request_output.csv', index=False)

#write taxi activities
Taxi_activities.to_csv('H:\\EAV Taxi Data\\'+scenario+'\\'+'taxi_activities.csv', index=False)

#write running time
plt.hist(deep_learning_times)    
sum(deep_learning_times)
pd.DataFrame(deep_learning_times).to_csv('H:\\EAV Taxi Data\\'+scenario+'\\deep_learning_times.csv', index=False)

plt.hist(optimization_dispatch_times)    
sum(optimization_dispatch_times)
pd.DataFrame(optimization_dispatch_times).to_csv('H:\\EAV Taxi Data\\'+scenario+'\\optimization_dispatch_times.csv', index=False)

plt.hist(optimization_times)    
sum(optimization_times)
pd.DataFrame(optimization_times).to_csv('H:\\EAV Taxi Data\\'+scenario+'\\optimization_times.csv', index=False)

