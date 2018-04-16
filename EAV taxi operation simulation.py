# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:38:41 2017

@author: lianghu

This file runs simulation and optimization

This file reads/writes taxi status csv in real time in dictionary

500 taxis take 3h. 1500 taxis take 10h.
"""

import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum, tuplelist
from geopy.distance import great_circle
import math
import itertools
from copy import deepcopy
from datetime import datetime
#from UpdateTaxiStatus_Dict import *

#start time
start_running = datetime.now() 
print start_running

#set up parameters
EV_range = 200.0 #miles
speed_average_NYC = 13.0 #mph
electricity_consumption_rate = 0.3 #kWh/mi
#charge_to = 1 #charge to 100%
need_charge = 0.1 #SOC < 10% range
time_interval = 60 #seconds; 1 min
pickup_ub = 60 #maximum pickup time



#UpdateTaxiStatus functions
taxi_activity = {} #create empty dict to store variable names
def CreateNewTaxiCSV(taxi): 
    for i in range(taxi.shape[0]):
        each_taxi = taxi.loc[i]
        each_taxi = pd.DataFrame(each_taxi).transpose()
        each_taxi = each_taxi.reset_index(drop=True)
        each_taxi.columns = ['medallion', 'time_start', 'start_longitude', 
                             'start_latitude', 'SOC_start']
        each_taxi.loc[0, 'time_end'] = each_taxi.loc[0, 'time_start']
        each_taxi.loc[0, 'time_duration_in_mins'] = 0
        each_taxi.loc[0, 'travel_distance'] = 0.0
        each_taxi.loc[0, 'end_longitude'] = each_taxi.loc[0, 'start_longitude']
        each_taxi.loc[0, 'end_latitude'] = each_taxi.loc[0, 'start_latitude']
        each_taxi.loc[0, 'SOC_end'] = each_taxi.loc[0, 'SOC_start']
        each_taxi.loc[0, 'status'] = 'start'
        #update dictionary
        taxi_activity[str(int(each_taxi.loc[0, 'medallion']))] = each_taxi

def UpdateTaxiStatusCalled(medallion, 
                           pickup_longitude, 
                           pickup_latitude,
                           travel_distance,
                           travel_time,
                           called_datetime,
                           arrive_datetime):
    t = str(int(medallion))
    #index of a new line
    n = taxi_activity[t].shape[0]
    #update n-th row
    taxi_activity[t].loc[n, 'medallion'] = medallion
    taxi_activity[t].loc[n, 'time_start'] = called_datetime
    taxi_activity[t].loc[n, 'start_longitude'] = taxi_activity[t].loc[n-1, 'end_longitude']
    taxi_activity[t].loc[n, 'start_latitude'] = taxi_activity[t].loc[n-1, 'end_latitude']
    taxi_activity[t].loc[n, 'SOC_start'] = taxi_activity[t].loc[n-1, 'SOC_end']
    taxi_activity[t].loc[n, 'time_end'] = arrive_datetime
    taxi_activity[t].loc[n, 'time_duration_in_mins'] = travel_time
    taxi_activity[t].loc[n, 'travel_distance'] = travel_distance
    taxi_activity[t].loc[n, 'end_longitude'] = pickup_longitude
    taxi_activity[t].loc[n, 'end_latitude'] = pickup_latitude
    taxi_activity[t].loc[n, 'SOC_end'] = taxi_activity[t].loc[n, 'SOC_start']-taxi_activity[t].loc[n, 'travel_distance']
    taxi_activity[t].loc[n, 'status'] = 'called'
      
def UpdateTaxiStatusOccupied(medallion,
                             pickup_datetime_actual,
                             dropoff_datetime_actual,
                             dropoff_longitude, 
                             dropoff_latitude,
                             travel_distance, 
                             travel_time):
    t = str(int(medallion))
    #index of a new line
    n = taxi_activity[t].shape[0]
    #update n-th row
    taxi_activity[t].loc[n, 'medallion'] = medallion
    taxi_activity[t].loc[n, 'time_start'] = pickup_datetime_actual
    taxi_activity[t].loc[n, 'start_longitude'] = taxi_activity[t].loc[n-1, 'end_longitude']
    taxi_activity[t].loc[n, 'start_latitude'] = taxi_activity[t].loc[n-1, 'end_latitude']
    taxi_activity[t].loc[n, 'SOC_start'] = taxi_activity[t].loc[n-1, 'SOC_end']
    taxi_activity[t].loc[n, 'time_end'] = dropoff_datetime_actual
    taxi_activity[t].loc[n, 'time_duration_in_mins'] = travel_time
    taxi_activity[t].loc[n, 'travel_distance'] = travel_distance
    taxi_activity[t].loc[n, 'end_longitude'] = dropoff_longitude
    taxi_activity[t].loc[n, 'end_latitude'] = dropoff_latitude
    taxi_activity[t].loc[n, 'SOC_end'] = taxi_activity[t].loc[n, 'SOC_start']-taxi_activity[t].loc[n, 'travel_distance']
    taxi_activity[t].loc[n, 'status'] = 'occupied'

def UpdateTaxiStatusWaiting(medallion):
    t = str(int(medallion))
    #index of a new line
    n = taxi_activity[t].shape[0]
    #update n-th row
    taxi_activity[t].loc[n, 'medallion'] = medallion
    taxi_activity[t].loc[n, 'time_start'] = taxi_activity[t].loc[n-1, 'time_end']
    taxi_activity[t].loc[n, 'start_longitude'] = taxi_activity[t].loc[n-1, 'end_longitude']
    taxi_activity[t].loc[n, 'start_latitude'] = taxi_activity[t].loc[n-1, 'end_latitude']
    taxi_activity[t].loc[n, 'SOC_start'] = taxi_activity[t].loc[n-1, 'SOC_end']
    taxi_activity[t].loc[n, 'time_end'] = taxi_activity[t].loc[n, 'time_start'] #need updated when the csv is finalized
    taxi_activity[t].loc[n, 'time_duration_in_mins'] = 0 #need updated when the csv is finalized
    taxi_activity[t].loc[n, 'travel_distance'] = 0.0
    taxi_activity[t].loc[n, 'end_longitude'] = taxi_activity[t].loc[n, 'start_longitude']
    taxi_activity[t].loc[n, 'end_latitude'] = taxi_activity[t].loc[n, 'start_latitude']
    taxi_activity[t].loc[n, 'SOC_end'] = taxi_activity[t].loc[n, 'SOC_start']
    taxi_activity[t].loc[n, 'status'] = 'waiting'

def UpdateTaxiStatusGoToCharging(medallion, 
                                 CS_longitude, 
                                 CS_latitude,
                                 dropoff_datetime_actual,
                                 travel_time,
                                 travel_distance):
    t = str(int(medallion))
    #index of a new line
    n = taxi_activity[t].shape[0]
    #update n-th row
    taxi_activity[t].loc[n, 'medallion'] = medallion
    taxi_activity[t].loc[n, 'time_start'] = dropoff_datetime_actual
    taxi_activity[t].loc[n, 'start_longitude'] = taxi_activity[t].loc[n-1, 'end_longitude']
    taxi_activity[t].loc[n, 'start_latitude'] = taxi_activity[t].loc[n-1, 'end_latitude']
    taxi_activity[t].loc[n, 'SOC_start'] = taxi_activity[t].loc[n-1, 'SOC_end']
    taxi_activity[t].loc[n, 'time_duration_in_mins'] = travel_time
    taxi_activity[t].loc[n, 'time_end'] = taxi_activity[t].loc[n, 'time_start']+travel_time*60
    taxi_activity[t].loc[n, 'travel_distance'] = travel_distance
    taxi_activity[t].loc[n, 'end_longitude'] = CS_longitude
    taxi_activity[t].loc[n, 'end_latitude'] = CS_latitude
    taxi_activity[t].loc[n, 'SOC_end'] = taxi_activity[t].loc[n, 'SOC_start']-taxi_activity[t].loc[n, 'travel_distance']
    taxi_activity[t].loc[n, 'status'] = 'go to charging'
  
def UpdateTaxiStatusCharging(medallion,
                             charge_duration,
                             finish_datetime):
    t = str(int(medallion))
    #index of a new line
    n = taxi_activity[t].shape[0]
    #update n-th row
    taxi_activity[t].loc[n, 'medallion'] = medallion
    taxi_activity[t].loc[n, 'start_longitude'] = taxi_activity[t].loc[n-1, 'end_longitude']
    taxi_activity[t].loc[n, 'start_latitude'] = taxi_activity[t].loc[n-1, 'end_latitude']
    taxi_activity[t].loc[n, 'SOC_start'] = taxi_activity[t].loc[n-1, 'SOC_end']
    taxi_activity[t].loc[n, 'time_end'] = finish_datetime
    taxi_activity[t].loc[n, 'time_duration_in_mins'] = charge_duration
    taxi_activity[t].loc[n, 'time_start'] = finish_datetime-charge_duration*60
    taxi_activity[t].loc[n, 'travel_distance'] = 0.0
    taxi_activity[t].loc[n, 'end_longitude'] = taxi_activity[t].loc[n, 'start_longitude']
    taxi_activity[t].loc[n, 'end_latitude'] = taxi_activity[t].loc[n, 'start_latitude']
    taxi_activity[t].loc[n, 'SOC_end'] = EV_range
    taxi_activity[t].loc[n, 'status'] = 'charging'

 

#read trip data
df = pd.read_csv('H:\\EAV Taxi Data\\Oct16_500taxis_rate.csv')
#change charging power to 50kW
df['CS_power'] = 50.0

#create taxi status table
medallion = sorted(list(set(df['medallion'])))
#cut fleet size to test
#np.random.seed(400) #seed=number of taxis
#medallion = sorted(list(np.random.choice(medallion, 400, replace=False)))
taxi = pd.DataFrame(medallion, columns=['medallion'])
taxi['available_datetime'] = ""
taxi['available_longitude'] = ""
taxi['available_latitude'] = ""
taxi['SOC'] = ""

#initiate taxi status table
for i in range(len(medallion)):
    each_taxi = df[df['medallion']==medallion[i]]
    each_taxi = each_taxi.sort_values(['pickup_datetime'])
    each_taxi = each_taxi.reset_index(drop=True)
    #taxi.loc[i, 'available_datetime'] = each_taxi.loc[0, 'pickup_datetime']
    taxi.loc[i, 'available_datetime'] = 1381896000
    taxi.loc[i, 'available_longitude'] = each_taxi.loc[0, 'pickup_longitude']
    taxi.loc[i, 'available_latitude'] = each_taxi.loc[0, 'pickup_latitude']
#randomly assign initial SOC [10%,100%] ~ uniform distribution
np.random.seed(1991)
taxi['SOC'] = EV_range*np.random.uniform(low=0.1, high=1.0, size=taxi.shape[0])
#create a csv for each taxi
CreateNewTaxiCSV(taxi)
#initial status of each taxi is waiting
for each_medallion in medallion:
    UpdateTaxiStatusWaiting(each_medallion)

#create and initiate request status table
request = df.loc[:, ['pickup_datetime', 'dropoff_datetime', 'trip_distance',
              'pickup_longitude', 'pickup_latitude',
              'dropoff_longitude', 'dropoff_latitude',
              'date', 'trip_time_in_mins', 'CS_longitude', 'CS_latitude',
              'CS_power', 'CS_distance', 'CS_travel_time_in_mins', 'demand_rate', 'supply_rate']]
request['pickup_datetime_actual'] = ""
request['pickup_delay_in_mins'] = 0
request['completed'] = "No"
request['taxi_serve'] = "" 

#create time interval list
start_time = 1381896000 #2013-10-16 00:00:00
end_time = 1381982400 #2013-10-17 00:00:00
time_range = range(start_time, end_time, time_interval)

for j in range(len(time_range)):
    
    if (j%20==0):
        print(j)
    
    #subset
    taxi_sub = taxi[taxi['available_datetime']<(time_range[j]+time_interval)]
    request_sub = request[(request['completed']=='No') & 
                          (request['pickup_datetime']<(time_range[j]+time_interval))]

    #get index, as keys for dictionary
    taxi_sub_index = taxi_sub.index.tolist()
    request_sub_index = request_sub.index.tolist()
    path = tuplelist(list(itertools.product(taxi_sub_index, request_sub_index)))

    #if no taxi and no request
    if (len(taxi_sub_index)==0 and len(request_sub_index)==0):
        continue
    #if no request only
    if (len(taxi_sub_index)>0 and len(request_sub_index)==0):
        #taxis wait for a time interval, update
        for each_waiting_taxi in taxi_sub_index:
            UpdateTaxiStatusWaiting(taxi_sub.loc[each_waiting_taxi, 'medallion'])
        continue
    #if no taxi only
    if (len(taxi_sub_index)==0 and len(request_sub_index)>0):
        request_sub['pickup_delay_in_mins'] +=  time_interval/60
        request.update(request_sub)
        continue
    
    #convert dataframe into dictionary
    taxi_sub = taxi_sub.to_dict(orient="index")
    request_sub = request_sub.to_dict(orient="index")

    #calculate pickup distance dictionary
    pickup_distance = pd.DataFrame(index=taxi_sub_index, columns=request_sub_index)
    pickup_distance = pickup_distance.to_dict(orient="index")
    for each_taxi in taxi_sub_index:
        for each_request in request_sub_index:
            taxi_GPS = (taxi_sub[each_taxi]['available_longitude'], taxi_sub[each_taxi]['available_latitude'])
            request_GPS = (request_sub[each_request]['pickup_longitude'], request_sub[each_request]['pickup_latitude'])
            pickup_distance[each_taxi][each_request] = 1.4413*great_circle(taxi_GPS, request_GPS).miles + 0.1383 #miles

    #calculate pickup time dictionary
    pickup_time_in_mins = deepcopy(pickup_distance)
    for each_taxi in taxi_sub_index:
        for each_request in request_sub_index:
            pickup_time_in_mins[each_taxi][each_request] = int(math.ceil(
                    pickup_time_in_mins[each_taxi][each_request]/speed_average_NYC*60))

    #calculate pickup delay time dictionary
    delay_time_in_mins = deepcopy(pickup_time_in_mins)
    for each_taxi in taxi_sub_index:
        for each_request in request_sub_index:
            delay_time_in_mins[each_taxi][each_request] = (pickup_time_in_mins[each_taxi][each_request]
            + request_sub[each_request]['pickup_delay_in_mins'])

    #calculate pickup distance + occupied distance dictionary
    pickup_occupied_distance = deepcopy(pickup_distance)
    for each_taxi in taxi_sub_index:
        for each_request in request_sub_index:
            pickup_occupied_distance[each_taxi][each_request] = (pickup_distance[each_taxi][each_request]
            + request_sub[each_request]['trip_distance'])
            
    #calculate pickup distance + occupied distance + charging distance dictionary
    pickup_occupied_CS_distance = deepcopy(pickup_distance)
    for each_taxi in taxi_sub_index:
        for each_request in request_sub_index:
            pickup_occupied_CS_distance[each_taxi][each_request] = (pickup_distance[each_taxi][each_request]
            + request_sub[each_request]['trip_distance']
            + request_sub[each_request]['CS_distance'])
     
    ### optimization starts from here ###
    """    
    #count amount of available taxis
    n_available_taxi = len(taxi_sub_index)
    taxi_sub_index_available = taxi_sub_index[:]
    for each_taxi in taxi_sub_index:
        if (min(pickup_occupied_CS_distance[each_taxi].values())>taxi_sub[each_taxi]['SOC']):
            n_available_taxi = n_available_taxi-1
            taxi_sub_index_available.remove(each_taxi) #get index of available taxis
    #count amount of requests
    n_request = len(request_sub_index)
    
    request_sub_index_near = request_sub_index[:]
    for each_request in request_sub_index:
        min_pickup = 2000 #minutes
        for each_taxi in taxi_sub_index:
            min_pickup = min(min_pickup, pickup_time_in_mins[each_taxi][each_request])
        if (min_pickup>pickup_ub):
            n_request_near = n_request_near-1
            request_sub_index_near.remove(each_request)
    
    #request w/delay >= 10 minutes, must be picked up
    #request_sub_index_delay = [each_request for each_request in request_sub_index if request_sub[each_request]['pickup_delay_in_mins']>=5]
    #request_sub_index_no_delay = [each_request for each_request in request_sub_index if request_sub[each_request]['pickup_delay_in_mins']<5]
    """

    #OPTIMIZATION
    #create optimization model
    model = Model()
    model.setParam('OutputFlag', 0)
    #add variables
    x = model.addVars(taxi_sub_index, request_sub_index, vtype=GRB.BINARY, name="x")
    #set objective
    #obj = x.prod(delay_time_in_mins)
    obj = quicksum(x[i,j]*delay_time_in_mins[i][j] for i,j in path) + quicksum((1-x.sum('*', j))*(request_sub[j]['pickup_delay_in_mins']+pickup_ub) for j in request_sub_index)
    model.setObjective(obj, GRB.MINIMIZE)
    #add constraints
    for each_taxi in taxi_sub_index:
        for each_request in request_sub_index:
            if (pickup_occupied_CS_distance[each_taxi][each_request]>taxi_sub[each_taxi]['SOC']):
                model.addConstr(x[each_taxi, each_request]==0) #taxi does not accept request if range is not enough
            #if (pickup_time_in_mins[each_taxi][each_request]>pickup_ub):
            #    model.addConstr(x[each_taxi, each_request]==0) #taxi does not accept request if too far 
    """       
    if (n_available_taxi==n_request):
        model.addConstrs(x.sum(i, '*')==1 for i in taxi_sub_index_available)
        model.addConstrs(x.sum('*', j)==1 for j in request_sub_index)
    if (n_available_taxi>n_request):
        model.addConstrs(x.sum(i, '*')<=1 for i in taxi_sub_index_available)
        model.addConstrs(x.sum('*', j)==1 for j in request_sub_index)
    if (n_available_taxi<n_request):
        model.addConstrs(x.sum(i, '*')==1 for i in taxi_sub_index_available)
        model.addConstrs(x.sum('*', j)<=1 for j in request_sub_index)
        #model.addConstrs(x.sum('*', j)<=1 for j in request_sub_index_no_delay)
        #model.addConstrs(x.sum('*', j)==1 for j in request_sub_index_delay)
    """
    model.addConstrs(x.sum(i, '*')<=1 for i in taxi_sub_index)
    model.addConstrs(x.sum('*', j)<=1 for j in request_sub_index)
    
    #solve
    model.optimize()
    #print solution
    #for v in model.getVars():
    #    print(v.varName, v.x)
    #print('Obj:', model.objVal)
    """
    #if no solutions
    if (model.SolCount==0):
        model = Model()
        model.setParam('OutputFlag', 0)
        x = model.addVars(taxi_sub_index, request_sub_index, vtype=GRB.BINARY, name="x")
        obj = quicksum(x[i,j]*delay_time_in_mins[i][j] for i,j in path) + quicksum((1-x.sum('*', j))*(request_sub[j]['pickup_delay_in_mins']+big_M) for j in request_sub_index)
        model.setObjective(obj, GRB.MINIMIZE)
        for each_taxi in taxi_sub_index:
            for each_request in request_sub_index:
                if (pickup_occupied_CS_distance[each_taxi][each_request]>taxi_sub[each_taxi]['SOC']):
                    model.addConstr(x[each_taxi, each_request]==0) #taxi does not accept request if range is not enough
                if (pickup_time_in_mins[each_taxi][each_request]>pickup_ub):
                    model.addConstr(x[each_taxi, each_request]==0) #taxi does not accept request if too far 
        model.addConstrs(x.sum(i, '*')<=1 for i in taxi_sub_index_available)
        model.addConstrs(x.sum('*', j)<=1 for j in request_sub_index_near)
        model.optimize()
    """    
    #convert the solved decision variables into a dictionary
    v = model.getAttr("x", model.getVars())
    v = map(int, v)
    v = np.array(v).reshape(len(taxi_sub_index), len(request_sub_index))
    v = pd.DataFrame(v, index=taxi_sub_index, columns=request_sub_index)
    v = v.to_dict(orient="index")
    
    
    #get one optimal solution
    if (j==540):
    #if (len(request_sub_index)>len(taxi_sub_index)):
        v_sub = pd.DataFrame.from_dict(v, orient='index')
        v_sub.to_csv('H:\\EAV Taxi Data\\solution\\v_sub.csv')
        taxi_sub_s = pd.DataFrame.from_dict(taxi_sub, orient='index')
        taxi_sub_s.to_csv('H:\\EAV Taxi Data\\solution\\taxi_sub.csv')
        request_sub_s = pd.DataFrame.from_dict(request_sub, orient='index')
        request_sub_s.to_csv('H:\\EAV Taxi Data\\solution\\request_sub.csv')
        break
    

    #update taxi status and request status
    #(1) when taxi-request match
    for each_taxi in taxi_sub_index:
        for each_request in request_sub_index:
            if (v[each_taxi][each_request]==1):
                #update request
                request_sub[each_request]['completed'] = 'Yes'
                request_sub[each_request]['taxi_serve'] = taxi_sub[each_taxi]['medallion']
                request_sub[each_request]['pickup_delay_in_mins'] = delay_time_in_mins[each_taxi][each_request]
                request_sub[each_request]['pickup_datetime_actual'] = (request_sub[each_request]['pickup_datetime'] 
                + request_sub[each_request]['pickup_delay_in_mins']*60)
                #update taxi
                UpdateTaxiStatusCalled(medallion=taxi_sub[each_taxi]['medallion'], 
                                       pickup_longitude=request_sub[each_request]['pickup_longitude'], 
                                       pickup_latitude=request_sub[each_request]['pickup_latitude'],
                                       travel_distance=pickup_distance[each_taxi][each_request],
                                       travel_time=pickup_time_in_mins[each_taxi][each_request],
                                       called_datetime=request_sub[each_request]['pickup_datetime'],
                                       arrive_datetime=request_sub[each_request]['pickup_datetime_actual'])
                UpdateTaxiStatusOccupied(medallion=taxi_sub[each_taxi]['medallion'],
                                         pickup_datetime_actual=request_sub[each_request]['pickup_datetime_actual'],
                                         dropoff_datetime_actual=(request_sub[each_request]['pickup_datetime_actual']+request_sub[each_request]['trip_time_in_mins']*60),
                                         dropoff_longitude=request_sub[each_request]['dropoff_longitude'], 
                                         dropoff_latitude=request_sub[each_request]['dropoff_latitude'],
                                         travel_distance=request_sub[each_request]['trip_distance'], 
                                         travel_time=request_sub[each_request]['trip_time_in_mins'])
                taxi_sub[each_taxi]['SOC'] = (taxi_sub[each_taxi]['SOC']
                -pickup_occupied_distance[each_taxi][each_request])
                if (taxi_sub[each_taxi]['SOC']>=need_charge*EV_range): #no need to charge
                    taxi_sub[each_taxi]['available_datetime'] = (request_sub[each_request]['pickup_datetime_actual'] + 
                            request_sub[each_request]['trip_time_in_mins']*60)
                    taxi_sub[each_taxi]['available_longitude'] = request_sub[each_request]['dropoff_longitude']
                    taxi_sub[each_taxi]['available_latitude'] = request_sub[each_request]['dropoff_latitude']
                if (taxi_sub[each_taxi]['SOC']<need_charge*EV_range): #need to charge
                    UpdateTaxiStatusGoToCharging(medallion=taxi_sub[each_taxi]['medallion'], 
                                                 CS_longitude=request_sub[each_request]['CS_longitude'], 
                                                 CS_latitude=request_sub[each_request]['CS_latitude'],
                                                 dropoff_datetime_actual=(request_sub[each_request]['pickup_datetime_actual']+request_sub[each_request]['trip_time_in_mins']*60),
                                                 travel_time=request_sub[each_request]['CS_travel_time_in_mins'],
                                                 travel_distance=request_sub[each_request]['CS_distance'])
                    charging_time_in_mins = int(
                            math.ceil((EV_range-taxi_sub[each_taxi]['SOC']+
                                       request_sub[each_request]['CS_distance'])*electricity_consumption_rate/request_sub[each_request]['CS_power']*60))
                    taxi_sub[each_taxi]['available_datetime'] = (request_sub[each_request]['pickup_datetime_actual']
                    + request_sub[each_request]['trip_time_in_mins']*60 
                    + request_sub[each_request]['CS_travel_time_in_mins']*60
                    + charging_time_in_mins*60)
                    taxi_sub[each_taxi]['available_longitude'] = request_sub[each_request]['CS_longitude']
                    taxi_sub[each_taxi]['available_latitude'] = request_sub[each_request]['CS_latitude']
                    taxi_sub[each_taxi]['SOC'] = EV_range
                    UpdateTaxiStatusCharging(medallion=taxi_sub[each_taxi]['medallion'],
                                             charge_duration=charging_time_in_mins,
                                             finish_datetime=taxi_sub[each_taxi]['available_datetime'])
                UpdateTaxiStatusWaiting(taxi_sub[each_taxi]['medallion'])
    #(2) when taxi not called
    for each_taxi in taxi_sub_index:
        if (sum(v[each_taxi].values())==0):
            UpdateTaxiStatusWaiting(taxi_sub[each_taxi]['medallion'])
    #(3) when request not accepted
    for each_request in request_sub_index:              
        v_sum = 0
        for each_taxi in taxi_sub_index:
            v_sum = v_sum + v[each_taxi][each_request]          
        if (v_sum==0):
            request_sub[each_request]['pickup_delay_in_mins'] +=  time_interval/60
        
    #convert taxi_sub and request_sub dictionaries into dataframes
    #and write back to taxi and request
    taxi_sub = pd.DataFrame.from_dict(taxi_sub, orient='index')
    taxi_sub = taxi_sub[taxi.columns.tolist()]
    taxi.update(taxi_sub)
    request_sub = pd.DataFrame.from_dict(request_sub, orient='index')
    request_sub = request_sub[request.columns.tolist()]
    request.update(request_sub)

#end time
print datetime.now()-start_running

#write output
#taxi.to_csv('H:\\EAV Taxi Data\\Oct16_500taxis_taxi_15.csv', index=False)
#request.to_csv('H:\\EAV Taxi Data\\Oct16_500taxis_request_15.csv', index=False)

#write taxi activities
#for key,df in taxi_activity.items():
#    df.to_csv('H:\\EAV Taxi Data\\Taxi\\'+key+'.csv', index=False)
    
    









