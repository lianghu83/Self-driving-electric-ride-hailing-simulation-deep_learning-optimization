"""
This file measures efficiency of old taxi system.
"""

import pandas as pd
from geopy.distance import great_circle
import numpy as np

df = pd.read_csv('H:\\EAV Taxi Data\\Raw Data\\10_16_taxis_5%.csv')

#create taxi_system dataframe
medallion = sorted(list(set(df['medallion'])))
taxi_system = pd.DataFrame(medallion, columns=['medallion'])
taxi_system['occupied_trips'] = ""
taxi_system['working_time_in_mins_total'] = "" #from 1st pickup to last dropoff
taxi_system['working_time_in_mins_occupied'] = ""
taxi_system['working_time_in_mins_empty'] = ""
taxi_system['travel_distance_total'] = ""
taxi_system['travel_distance_occupied'] = ""
taxi_system['travel_distance_empty'] = ""

#function calculates empty time
def CalculateEmptyTime(df):
    empty_time_total = 0
    for i in range(df.shape[0]-1):
        empty_time = df.loc[i+1, 'pickup_datetime'] - df.loc[i, 'dropoff_datetime']
        empty_time_total = empty_time_total + empty_time
    empty_time_total = empty_time_total / 60 #minutes
    return(empty_time_total)

#function calculates empty distance
def CalculateEmptyDistance(df):
    empty_distance_total = 0.0
    for i in range(df.shape[0]-1):
        dropoff = (df.loc[i, 'dropoff_latitude'], df.loc[i, 'dropoff_longitude'])
        pickup = (df.loc[i+1, 'pickup_latitude'], df.loc[i+1, 'pickup_longitude'])
        empty_distance = 1.4413*great_circle(dropoff, pickup).miles+0.1383 #miles
        empty_distance_total += empty_distance
    return(empty_distance_total)

#iterate each taxi
for i in range(len(medallion)):
    df_each_taxi = df[df['medallion']==medallion[i]]
    df_each_taxi = df_each_taxi.sort_values(['pickup_datetime'])
    df_each_taxi = df_each_taxi.reset_index(drop=True)
    #number of occupied trips
    taxi_system.loc[i, 'occupied_trips'] = df_each_taxi.shape[0]
    #time
    taxi_system.loc[i, 'working_time_in_mins_total'] = (
            max(df_each_taxi['dropoff_datetime']) - 
            min(df_each_taxi['pickup_datetime'])) / 60
    taxi_system.loc[i, 'working_time_in_mins_occupied'] = sum(df_each_taxi['trip_time_in_mins'])
    taxi_system.loc[i, 'working_time_in_mins_empty'] = CalculateEmptyTime(df_each_taxi)
    #distance
    taxi_system.loc[i, 'travel_distance_occupied'] = sum(df_each_taxi['trip_distance'])
    taxi_system.loc[i, 'travel_distance_empty'] = CalculateEmptyDistance(df_each_taxi)
    taxi_system.loc[i, 'travel_distance_total'] = (taxi_system.loc[i, 'travel_distance_occupied']
               + taxi_system.loc[i, 'travel_distance_empty'])

#write taxi_system
taxi_system.to_csv('H:\\EAV Taxi Data\\'+'10_16_op'+'\\old_system_output.csv', index=False)

#efficieny of taxi system
taxi_system['occupied_trips'] = taxi_system['occupied_trips'].astype(float)
taxi_system['occupied_trips'].describe()
np.percentile(taxi_system['occupied_trips'], 90)
np.percentile(taxi_system['occupied_trips'], 95)
np.percentile(taxi_system['occupied_trips'], 99) 

taxi_system['travel_distance_total'] = taxi_system['travel_distance_total'].astype(float)
taxi_system['travel_distance_total'].describe()
np.percentile(taxi_system['travel_distance_total'], 90)
np.percentile(taxi_system['travel_distance_total'], 95)
np.percentile(taxi_system['travel_distance_total'], 99) 

taxi_system['occupied_distance_ratio'] = taxi_system['travel_distance_occupied'] / taxi_system['travel_distance_total']
taxi_system['occupied_distance_ratio'] = taxi_system['occupied_distance_ratio'].astype(float)
taxi_system['occupied_distance_ratio'].describe()
np.percentile(taxi_system['occupied_distance_ratio'], 90)
np.percentile(taxi_system['occupied_distance_ratio'], 95)
np.percentile(taxi_system['occupied_distance_ratio'], 99) 

taxi_system['travel_distance_occupied'] = taxi_system['travel_distance_occupied'].astype(float)
taxi_system['travel_distance_occupied'].describe()
np.percentile(taxi_system['travel_distance_occupied'], 90)
np.percentile(taxi_system['travel_distance_occupied'], 95)
np.percentile(taxi_system['travel_distance_occupied'], 99) 
       
taxi_system['working_time_ratio'] = 1.0*taxi_system['working_time_in_mins_total'] / 1440
taxi_system['working_time_ratio'] = taxi_system['working_time_ratio'].astype(float)
taxi_system['working_time_ratio'].describe() 
np.percentile(taxi_system['working_time_ratio'], 90)
np.percentile(taxi_system['working_time_ratio'], 95)
np.percentile(taxi_system['working_time_ratio'], 99) 

taxi_system['occupied_time_ratio'] = 1.0*taxi_system['working_time_in_mins_occupied'] / taxi_system['working_time_in_mins_total']
taxi_system['occupied_time_ratio'] = taxi_system['occupied_time_ratio'].astype(float)
taxi_system['occupied_time_ratio'].describe()
np.percentile(taxi_system['occupied_time_ratio'], 90)
np.percentile(taxi_system['occupied_time_ratio'], 95)
np.percentile(taxi_system['occupied_time_ratio'], 99) 

taxi_system['occupied_time_ratio_24h'] = 1.0*taxi_system['working_time_in_mins_occupied'] / 1440
taxi_system['occupied_time_ratio_24h'] = taxi_system['occupied_time_ratio_24h'].astype(float)
taxi_system['occupied_time_ratio_24h'].describe()
np.percentile(taxi_system['occupied_time_ratio_24h'], 90)
np.percentile(taxi_system['occupied_time_ratio_24h'], 95)
np.percentile(taxi_system['occupied_time_ratio_24h'], 99) 
