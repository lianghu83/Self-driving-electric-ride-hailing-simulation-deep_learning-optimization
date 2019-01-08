"""
This file measures efficiency of new taxi system.
This file deals with the simulation/optimization results.
"""

import pandas as pd
#from os import listdir
import numpy as np
from Parameters import start_time, end_time, scenario, charging_power, electricity_consumption_rate, EV_range, time_range

#%%modify Taxi_activities
dir_path = 'H:\\EAV Taxi Data\\'+scenario
Taxi_activities = pd.read_csv(dir_path+'\\'+'taxi_activities.csv')
medallion = np.unique(Taxi_activities['medallion']).tolist()
medallion = sorted(medallion)

for i in range(len(medallion)):
    df = Taxi_activities[Taxi_activities['medallion'] == medallion[i]]
    df_index = df.index
    df = df.reset_index(drop=True)
    #fix end_time and time_duration for waiting status
    for j in range(df.shape[0]-1):
        if (df.loc[j, 'status'] in ['waiting', 'start charging']):
            df.loc[j, 'end_timestamp'] = df.loc[j+1, 'start_timestamp']
            df.loc[j, 'status_time'] = int((df.loc[j, 'end_timestamp']-df.loc[j, 'start_timestamp'])/60)
            df.loc[j, 'end_range'] = df.loc[j+1, 'start_range']
    #set max timestamp as end_time
    #df = df[df['start_timestamp']<=end_time]
    df['end_timestamp'].iloc[-1] = end_time
    df['status_time'].iloc[-1] = int((df['end_timestamp'].iloc[-1]-df['start_timestamp'].iloc[-1])/60)
    if df['status'].iloc[-1]=='start charging':
        add_range = min(df['status_time'].iloc[-1]/60*charging_power/electricity_consumption_rate, EV_range-df['start_range'].iloc[-1])
        df['end_range'].iloc[-1] = df['start_range'].iloc[-1] + add_range
    df.index = df_index
    Taxi_activities.update(df)

#write taxi acticitiy
Taxi_activities.to_csv('H:\\EAV Taxi Data\\'+scenario+'\\'+'taxi_activities.csv', index=False)

#%%create taxi_system dataframe
taxi_system = pd.DataFrame(medallion, columns=['medallion'])
taxi_system['occupied_trips'] = ""
taxi_system['working_time_in_mins_total'] = "" #from 1st pickup to last dropoff
taxi_system['working_time_in_mins_occupied'] = ""
taxi_system['working_time_in_mins_empty'] = ""
taxi_system['travel_distance_total'] = ""
taxi_system['travel_distance_occupied'] = ""
taxi_system['travel_distance_empty'] = ""
taxi_system['go_charging_time'] =  ""
taxi_system['charging_time'] =  ""
taxi_system['charging_number'] =  ""

#iterate over each taxi
for i in range(len(medallion)):
    df = Taxi_activities[Taxi_activities['medallion'] == medallion[i]]
    #set max timestamp as end_time
    df = df[df['start_timestamp']<=end_time]
    
    #number of occupied trips
    taxi_system.loc[i, 'occupied_trips'] = df[df['status']=='occupied'].shape[0]
    #time
    #taxi_system.loc[i, 'working_time_in_mins_total'] = int((
    #        max(df['time_end']) - 
    #        min(df['time_start'])) / 60)
    taxi_system.loc[i, 'working_time_in_mins_total'] = int((
            max(df[df['status']=='occupied']['end_timestamp']) - 
            min(df['start_timestamp'])) / 60) #working time=last dropoff-initiation
    taxi_system.loc[i, 'working_time_in_mins_occupied'] = int(sum(df[df['status']=='occupied']['status_time']))
    taxi_system.loc[i, 'working_time_in_mins_empty'] = taxi_system.loc[i, 'working_time_in_mins_total']-taxi_system.loc[i, 'working_time_in_mins_occupied']
    #distance
    taxi_system.loc[i, 'travel_distance_total'] = sum(df['status_distance'])
    taxi_system.loc[i, 'travel_distance_occupied'] = sum(df[df['status']=='occupied']['status_distance'])
    taxi_system.loc[i, 'travel_distance_empty'] = taxi_system.loc[i, 'travel_distance_total']-taxi_system.loc[i, 'travel_distance_occupied']
    taxi_system.loc[i, 'go_charging_time'] = sum(df[df['status']=='go charging']['status_time'])
    taxi_system.loc[i, 'charging_time'] = sum(df[df['status']=='start charging']['status_time'])
    taxi_system.loc[i, 'charging_number'] =  df[df['status']=='go charging'].shape[0]
 
#write taxi_system
taxi_system.to_csv('H:\\EAV Taxi Data\\'+scenario+'\\system_output.csv', index=False)
#read taxi_system
#taxi_system = pd.read_csv('H:\\EAV Taxi Data\\'+scenario+'\\system_output.csv')

#%%efficieny of taxi system
taxi_system['occupied_trips'] = taxi_system['occupied_trips'].astype(float)
taxi_system['occupied_trips'].describe()
np.percentile(taxi_system['occupied_trips'], 90)

taxi_system['travel_distance_total'] = taxi_system['travel_distance_total'].astype(float)
taxi_system['travel_distance_total'].describe()
np.percentile(taxi_system['travel_distance_total'], 90)
sum(taxi_system['travel_distance_total'])

taxi_system['occupied_distance_ratio'] = taxi_system['travel_distance_occupied'] / taxi_system['travel_distance_total']
taxi_system['occupied_distance_ratio'] = taxi_system['occupied_distance_ratio'].astype(float)
taxi_system['occupied_distance_ratio'].describe()
np.percentile(taxi_system['occupied_distance_ratio'], 90)

taxi_system['travel_distance_occupied'] = taxi_system['travel_distance_occupied'].astype(float)
taxi_system['travel_distance_occupied'].describe()
np.percentile(taxi_system['travel_distance_occupied'], 90)

taxi_system['travel_distance_empty'] = taxi_system['travel_distance_empty'].astype(float)
taxi_system['travel_distance_empty'].describe()
np.percentile(taxi_system['travel_distance_empty'], 90)
       
taxi_system['working_time_ratio'] = 1.0*taxi_system['working_time_in_mins_total'] / 1440
taxi_system['working_time_ratio'] = taxi_system['working_time_ratio'].astype(float)
taxi_system['working_time_ratio'].describe() 
np.percentile(taxi_system['working_time_ratio'], 90)

taxi_system['occupied_time_ratio'] = 1.0*taxi_system['working_time_in_mins_occupied'] / taxi_system['working_time_in_mins_total']
taxi_system['occupied_time_ratio'] = taxi_system['occupied_time_ratio'].astype(float)
taxi_system['occupied_time_ratio'].describe()
np.percentile(taxi_system['occupied_time_ratio'], 90)

taxi_system['occupied_time_ratio_24h'] = 1.0*taxi_system['working_time_in_mins_occupied'] / 1440
taxi_system['occupied_time_ratio_24h'] = taxi_system['occupied_time_ratio_24h'].astype(float)
taxi_system['occupied_time_ratio_24h'].describe()
np.percentile(taxi_system['occupied_time_ratio_24h'], 90)

taxi_system['go_charging_time'] = taxi_system['go_charging_time'].astype(float)
taxi_system['go_charging_time'].describe() 

taxi_system['charging_time'] = taxi_system['charging_time'].astype(float)
taxi_system['charging_time'].describe() 

taxi_system['charging_number'].describe() 

#%%taxi status change over time
status_change = pd.DataFrame(list(time_range), columns=['time_range'])
np.unique(Taxi_activities['status'])
status_change['called'] = ""
status_change['go charging'] = ""
status_change['occupied'] = ""
status_change['start charging'] = ""
status_change['waiting'] = ""

for i in range(len(time_range)):
    df = Taxi_activities[(Taxi_activities['start_timestamp'] <= time_range[i]) & (time_range[i] < Taxi_activities['end_timestamp'])]
    status_change.loc[i, 'called'] = df[df['status']=='called'].shape[0]
    status_change.loc[i, 'go charging'] = df[df['status']=='go charging'].shape[0]
    status_change.loc[i, 'occupied'] = df[df['status']=='occupied'].shape[0]
    status_change.loc[i, 'start charging'] = df[df['status']=='start charging'].shape[0]
    status_change.loc[i, 'waiting'] = df[df['status']=='waiting'].shape[0]

status_change['minute'] = (status_change['time_range'] - start_time)/60

writer = pd.ExcelWriter('H:\\EAV Taxi Data\\'+scenario+'\\status_change.xlsx')
status_change.to_excel(writer,'Sheet1', index=False)
writer.save()





















