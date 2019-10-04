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
from sklearn.metrics import recall_score, precision_score, accuracy_score



#%%
#record running time
optimization_times = []
deep_learning_times = []

#%%initiate big request status table
#select dataset
data_used = "10_16_requests_100%"
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
np.random.seed(1992)
Request['wait_time'] = np.random.randint(0, 16, Request.shape[0])
Request['wait_time'].describe()
Request['served'] = False
Request['served_taxi'] = -1
Request['taxi_pickup_time'] = -1
Request['pickup_timestamp'] = -1
Request['dropoff_timestamp'] = -1
Request_col = Request.columns.tolist()
Request['cs_distance'].describe()



#%%initiate big taxi status table
data_used = "10_16_taxis_100%"
df = pd.read_csv('H:\\EAV Taxi Data\\Raw Data\\'+data_used+'.csv') 
#drop 2013000000
df['medallion'] = df['medallion'] % 2013000000
    
medallion = sorted(list(set(df['medallion'])))


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



#%%
taxi_sub = Taxi

taxi_sub_index = taxi_sub.index.tolist()

taxi_sub = taxi_sub.to_dict(orient="index")

#record model performance on entire fleet
acc = []
prec = []
recall = []

for j in np.array([0,3,6,9,12,15,18,21])*60: #change time here, np.array([0,3,6,9,12,15,18,21])*60
    print(j/60)
    
    ### Preparation for dispatch ###

    request_sub = Request[(Request['served']==False) & 
                          (Request['wait_time']<=max_wait_time) &
                          (Request['origin_timestamp']==time_range[j])]
    
    #get index, as keys for dictionary
    request_sub_index = request_sub.index.tolist()
    match_path = tuplelist(list(itertools.product(taxi_sub_index, request_sub_index)))
    
    #convert dataframe into dictionary
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
    
    #CENTRALIZED OPTIMIZATION
    start_running_1 = datetime.now()
    v_op = CentralizedOptimization3(taxi_sub,
                                 taxi_sub_index,
                                 request_sub,
                                 request_sub_index,
                                 match_path,
                                 pickup_distance,
                                 pickup_time_in_mins)
    optimization_times.append((datetime.now()-start_running_1).total_seconds())
       
    #MACHINE LEARNING
    X, df_temp = DeepLearningDispatchPrepareData(taxi_sub,
                                                 Taxi_col,
                                                 request_sub,
                                                 Request_col,                       
                                                 pickup_distance,
                                                 pickup_time_in_mins,
                                                 time_range[j])
    start_running_1 = datetime.now()
    v_ml = DeepLearningDispatch2(X, df_temp, taxi_sub_index, request_sub_index)
    deep_learning_times.append((datetime.now()-start_running_1).total_seconds())
    
    #model performance
    v_op = pd.DataFrame.from_dict(v_op).sort_index().transpose().sort_index().stack().reset_index(drop=True)
    v_ml = pd.DataFrame.from_dict(v_ml).sort_index().transpose().sort_index().stack().reset_index(drop=True)
    acc.append(accuracy_score(v_op, v_ml))
    prec.append(precision_score(v_op, v_ml))
    recall.append(recall_score(v_op, v_ml))
    
    
#%%print out times

optimization_times
pd.DataFrame(optimization_times).to_csv('C:\\EAV Taxi Data\\'+'10_16_100%_R1'+'\\optimization_times.csv', index=False)
deep_learning_times
pd.DataFrame(deep_learning_times).to_csv('C:\\EAV Taxi Data\\'+'10_16_100%_R1'+'\\deep_learning_times.csv', index=False)
acc
pd.DataFrame(acc).to_csv('C:\\EAV Taxi Data\\'+'10_16_100%_R1'+'\\acc.csv', index=False)
prec
pd.DataFrame(prec).to_csv('C:\\EAV Taxi Data\\'+'10_16_100%_R1'+'\\prec.csv', index=False)
recall
pd.DataFrame(recall).to_csv('C:\\EAV Taxi Data\\'+'10_16_100%_R1'+'\\recall.csv', index=False)



#%% Plot

optimization_times = pd.read_csv('C:\\EAV Taxi Data\\'+'10_16_100%_R1'+'\\optimization_times.csv')['0'].tolist()
deep_learning_times = pd.read_csv('C:\\EAV Taxi Data\\'+'10_16_100%_R1'+'\\deep_learning_times.csv')['0'].tolist()

fig = plt.figure(figsize=(5, 3))
N = 8
ind = np.arange(N) 
width = 0.3    
plt.bar(ind, optimization_times, width, label='Optimization model', color='b', alpha=0.5)
plt.bar(ind + width, deep_learning_times, width,
    label='ANN model', color='g', alpha=0.5)
plt.xlabel("Instance")
plt.ylabel('Computation time (sec)')
plt.xticks(ind + width / 2, ('0:00','3:00','6:00','9:00','12:00','15:00','18:00','21:00'))
plt.legend(loc='best')
plt.tight_layout()
plt.show()
fig.savefig("C:\\EAV Taxi Data\\"+'10_16_100%_R1'+"\\computation_time_100%"+".jpg", dpi=300)




