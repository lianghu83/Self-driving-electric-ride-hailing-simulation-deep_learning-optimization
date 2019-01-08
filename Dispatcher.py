"""
Define different dispatching rules
"""

import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
from Parameters import pickup_ub, low_range, max_pickup_dist, max_pickup_time, start_time
#from copy import deepcopy

#%%
def CentralizedOptimization(taxi_sub,
                            taxi_sub_index,
                            request_sub,
                            request_sub_index,
                            path,
                            delay_time_in_mins,
                            pickup_occupied_distance,
                            pickup_distance):
    
    #create optimization model
    model = Model()
    model.setParam('OutputFlag', 0)
    
    #add variables
    x = model.addVars(taxi_sub_index, request_sub_index, vtype=GRB.BINARY, name="x")
    
    #set objective
    obj = quicksum(x[i,j]*delay_time_in_mins[i][j] for i,j in path) + quicksum((1-x.sum('*', j))*(request_sub[j]['pickup_delay_in_mins']+pickup_ub) for j in request_sub_index)
    #model.setObjective(obj, GRB.MAXIMIZE)
    model.setObjective(obj, GRB.MINIMIZE)
    
    #add constraints
    for each_taxi in taxi_sub_index:
        for each_request in request_sub_index:
            remaining_range = taxi_sub[each_taxi]['SOC']-pickup_occupied_distance[each_taxi][each_request]
            if ((remaining_range<low_range) & (remaining_range<request_sub[each_request]['CS_distance'])):
                model.addConstr(x[each_taxi, each_request]==0) 
            if (pickup_distance[each_taxi][each_request]>max_pickup_dist):
                model.addConstr(x[each_taxi, each_request]==0)
            #if (pickup_occupied_CS_distance[each_taxi][each_request]>taxi_sub[each_taxi]['SOC']):
            #    model.addConstr(x[each_taxi, each_request]==0) #taxi does not accept request if range is not enough
            #if (pickup_time_in_mins[each_taxi][each_request]>pickup_ub):
            #    model.addConstr(x[each_taxi, each_request]==0) #taxi does not accept request if too far 
    model.addConstrs(x.sum(i, '*')<=1 for i in taxi_sub_index)
    model.addConstrs(x.sum('*', j)<=1 for j in request_sub_index)
    
    #solve
    model.optimize()

    #convert the solved decision variables into a dictionary
    v = model.getAttr("x", model.getVars())
    v = list(map(int, v))
    v = np.array(v).reshape(len(taxi_sub_index), len(request_sub_index))
    v = pd.DataFrame(v, index=taxi_sub_index, columns=request_sub_index)
    v = v.to_dict(orient="index")
    
    return v

#%%
def CentralizedOptimization2(taxi_sub,
                             taxi_sub_index,
                             request_sub,
                             request_sub_index,
                             path,
                             #delay_time_in_mins,
                             pickup_occupied_distance,
                             #pickup_distance,
                             pickup_time_in_mins):
    
    #create optimization model
    model = Model()
    model.setParam('OutputFlag', 0)
    
    #add variables
    x = model.addVars(taxi_sub_index, request_sub_index, vtype=GRB.BINARY, name="x")
    
    #set objective
    obj = quicksum(x[i,j]*pickup_time_in_mins[i][j] for i,j in path) + quicksum((1-x.sum('*', j))*(request_sub[j]['pickup_delay_in_mins']+max_pickup_time) for j in request_sub_index)
    #model.setObjective(obj, GRB.MAXIMIZE)
    model.setObjective(obj, GRB.MINIMIZE)
    
    #add constraints
    model.addConstrs(x.sum(i, '*')<=1 for i in taxi_sub_index)
    model.addConstrs(x.sum('*', j)<=1 for j in request_sub_index)
    model.addConstrs((taxi_sub[i]['SOC']-pickup_occupied_distance[i][j])*x[i,j]>=0 for i,j in path)
    model.addConstrs((max_pickup_time-pickup_time_in_mins[i][j])*x[i,j]>=0 for i,j in path)
    
    #solve
    model.optimize()

    #convert the solved decision variables into a dictionary
    v = model.getAttr("x", model.getVars())
    v = list(map(int, v))
    v = np.array(v).reshape(len(taxi_sub_index), len(request_sub_index))
    v = pd.DataFrame(v, index=taxi_sub_index, columns=request_sub_index)
    v = v.to_dict(orient="index")
    
    return v

#%%
def CentralizedOptimization3(taxi_sub,
                             taxi_sub_index,
                             request_sub,
                             request_sub_index,
                             path,
                             pickup_distance,
                             pickup_time_in_mins):
    
    #create optimization model
    model = Model()
    model.setParam('OutputFlag', 0)
    
    #add variables
    x = model.addVars(taxi_sub_index, request_sub_index, vtype=GRB.BINARY, name="x")
    
    #set objective
    obj = quicksum(x[i,j]*pickup_time_in_mins[i][j] for i,j in path) + quicksum((1-x.sum('*', j))*(request_sub[j]['wait_time']+max_pickup_time) for j in request_sub_index)
    model.setObjective(obj, GRB.MINIMIZE)
    
    #add constraints
    model.addConstrs(x.sum(i, '*')<=1 for i in taxi_sub_index)
    model.addConstrs(x.sum('*', j)<=1 for j in request_sub_index)
    model.addConstrs((taxi_sub[i]['end_range']-pickup_distance[i][j]-request_sub[j]['trip_distance'])*x[i,j]>=0 for i,j in path)
    model.addConstrs((max_pickup_time-pickup_time_in_mins[i][j])*x[i,j]>=0 for i,j in path)
    
    #solve
    model.optimize()

    #convert the solved decision variables into a dictionary
    v = model.getAttr("x", model.getVars())
    v = list(map(int, v))
    v = np.array(v).reshape(len(taxi_sub_index), len(request_sub_index))
    v = pd.DataFrame(v, index=taxi_sub_index, columns=request_sub_index)
    v = v.to_dict(orient="index")
    
    return v

#%%
def CentralizedOptimization4(taxi_sub,
                             taxi_sub_index,
                             request_sub,
                             request_sub_index,
                             path,
                             pickup_distance,
                             pickup_time_in_mins):
    
    #create optimization model
    model = Model()
    model.setParam('OutputFlag', 0)
    
    #add variables
    x = model.addVars(taxi_sub_index, request_sub_index, vtype=GRB.BINARY, name="x")
    
    #set objective
    obj = quicksum(x[i,j]*pickup_time_in_mins[i][j] for i,j in path) + quicksum((1-x.sum('*', j))*(request_sub[j]['wait_time']+max_pickup_time) for j in request_sub_index)
    model.setObjective(obj, GRB.MINIMIZE)
    
    #add constraints
    model.addConstrs(x.sum(i, '*')<=1 for i in taxi_sub_index)
    model.addConstrs(x.sum('*', j)<=1 for j in request_sub_index)
    model.addConstrs((taxi_sub[i]['end_range']-pickup_distance[i][j]-request_sub[j]['trip_distance'])*x[i,j]>=0 for i,j in path)
    model.addConstrs((request_sub[j]['wait_time']+max_pickup_time-pickup_time_in_mins[i][j])*x[i,j]>=0 for i,j in path)
    
    #solve
    model.optimize()

    #convert the solved decision variables into a dictionary
    v = model.getAttr("x", model.getVars())
    v = list(map(int, v))
    v = np.array(v).reshape(len(taxi_sub_index), len(request_sub_index))
    v = pd.DataFrame(v, index=taxi_sub_index, columns=request_sub_index)
    v = v.to_dict(orient="index")
    
    return v

#%%
def CombineTaxiRequest(taxi_sub1,
                       Taxi_col,
                       request_sub1,
                       Request_col,
                       pickup_distance,
                       pickup_time_in_mins,
                       v,
                       timestamp):

    taxi_sub1 = pd.DataFrame.from_dict(taxi_sub1, orient='index')
    taxi_sub1 = taxi_sub1[Taxi_col]
    request_sub1 = pd.DataFrame.from_dict(request_sub1, orient='index')
    request_sub1 = request_sub1[Request_col]
    #drop not useful columns from request_sub1
    request_sub1.drop(['served','served_taxi','taxi_pickup_time','pickup_timestamp','dropoff_timestamp'], axis=1, inplace=True)
    
    #join taxi and request
    taxi_sub_pop = pd.concat([taxi_sub1]*request_sub1.shape[0]).sort_index().reset_index(drop=True)
    request_sub1 = request_sub1.sort_index()
    request_sub_pop = pd.concat([request_sub1]*taxi_sub1.shape[0]).reset_index(drop=True)
    taxi_request_join = taxi_sub_pop.join(request_sub_pop)

    #add more variables from dictionaries
    taxi_request_join['pickup_distance'] = pd.DataFrame.from_dict(pickup_distance).sort_index().transpose().sort_index().stack().reset_index(drop=True)
    taxi_request_join['pickup_time_in_mins'] = pd.DataFrame.from_dict(pickup_time_in_mins).sort_index().transpose().sort_index().stack().reset_index(drop=True)
 
    #add current timestamp
    taxi_request_join['timestamp'] = timestamp
    
    #add decision variables as class
    taxi_request_join['class'] = pd.DataFrame.from_dict(v).sort_index().transpose().sort_index().stack().reset_index(drop=True)
    
    #drop too far taxi-request pairs
    #taxi_request_join = taxi_request_join[taxi_request_join['pickup_distance']<=max_pickup_dist]
    taxi_request_join = taxi_request_join[taxi_request_join['pickup_time_in_mins']<=max_pickup_time]
    
    #select useful columns for machine learning
    used_col = ['end_longitude', 'end_latitude', 'end_range',\
            'origin_longitude', 'origin_latitude', 'trip_distance', 'trip_time', 'wait_time',\
            'pickup_distance', 'pickup_time_in_mins',\
            'timestamp',\
            'class']
    taxi_request_join = taxi_request_join[used_col]
    
    return taxi_request_join

#%%
def DeepLearningDispatchPrepareData(taxi_sub1,
                                    Taxi_col,
                                    request_sub1,
                                    Request_col,                       
                                    pickup_distance,
                                    pickup_time_in_mins,
                                    timestamp):
    #1. join taxi and request data
    
    taxi_sub1 = pd.DataFrame.from_dict(taxi_sub1, orient='index')
    taxi_sub1 = taxi_sub1[Taxi_col]
    request_sub1 = pd.DataFrame.from_dict(request_sub1, orient='index')
    request_sub1 = request_sub1[Request_col]
    #drop not useful columns from request_sub1
    request_sub1.drop(['served','served_taxi','taxi_pickup_time','pickup_timestamp','dropoff_timestamp'], axis=1, inplace=True)
    
    #join taxi and request
    taxi_sub_pop = pd.concat([taxi_sub1]*request_sub1.shape[0]).sort_index().reset_index(drop=True)
    request_sub1 = request_sub1.sort_index()
    request_sub_pop = pd.concat([request_sub1]*taxi_sub1.shape[0]).reset_index(drop=True)
    taxi_request_join = taxi_sub_pop.join(request_sub_pop)

    #add more variables from dictionaries
    taxi_request_join['pickup_distance'] = pd.DataFrame.from_dict(pickup_distance).sort_index().transpose().sort_index().stack().reset_index(drop=True)
    taxi_request_join['pickup_time_in_mins'] = pd.DataFrame.from_dict(pickup_time_in_mins).sort_index().transpose().sort_index().stack().reset_index(drop=True)
 
    #add current timestamp
    taxi_request_join['timestamp'] = (timestamp-start_time)/60%1440
      
    #drop too far taxi-request pairs
    #taxi_request_join = taxi_request_join[taxi_request_join['pickup_distance']<=max_pickup_dist]
    taxi_request_join = taxi_request_join[taxi_request_join['pickup_time_in_mins']<=max_pickup_time]
    
    #enough energy
    taxi_request_join['enough_range'] = taxi_request_join['end_range'] - taxi_request_join['pickup_distance'] - taxi_request_join['trip_distance']
    taxi_request_join = taxi_request_join[taxi_request_join['enough_range']>=0]
    
    #2. prepare data for deep learning dispatch model
    used_col = ['end_longitude', 'end_latitude', 'end_range',\
                'origin_longitude', 'origin_latitude', 'trip_distance', 'trip_time', 'wait_time',\
                'pickup_distance', 'pickup_time_in_mins',\
                'timestamp']
    X = taxi_request_join[used_col]
    #d = {'waiting': 0, 'start charging': 1}
    #X['status'] = X['status'].map(d)
    
    #use the scaling results of the training samples
    X = sc.transform(X)
    
    df = taxi_request_join[['medallion', 'id']]
    
    return X, df

#%%

from keras.models import load_model
ml_model = load_model('H:\\EAV Taxi Data\\9_10-12'+'\\ann_model_128-64-8-256.h5')
import pickle
sc_pickle = pickle.load(open("H:\\EAV Taxi Data\\9_10-12"+"\\sc_pickle.p", "rb" ))
sc = sc_pickle["sc"]

#%%
def DeepLearningDispatch(taxi_sub1,
                         Taxi_col,
                         request_sub1,
                         Request_col,                       
                         pickup_distance,
                         pickup_time_in_mins,
                         timestamp):
    #1. join taxi and request data
    
    taxi_sub1 = pd.DataFrame.from_dict(taxi_sub1, orient='index')
    taxi_sub1 = taxi_sub1[Taxi_col]
    request_sub1 = pd.DataFrame.from_dict(request_sub1, orient='index')
    request_sub1 = request_sub1[Request_col]
    #drop not useful columns from request_sub1
    request_sub1.drop(['served','served_taxi','taxi_pickup_time','pickup_timestamp','dropoff_timestamp'], axis=1, inplace=True)
    
    #join taxi and request
    taxi_sub_pop = pd.concat([taxi_sub1]*request_sub1.shape[0]).sort_index().reset_index(drop=True)
    request_sub1 = request_sub1.sort_index()
    request_sub_pop = pd.concat([request_sub1]*taxi_sub1.shape[0]).reset_index(drop=True)
    taxi_request_join = taxi_sub_pop.join(request_sub_pop)

    #add more variables from dictionaries
    taxi_request_join['pickup_distance'] = pd.DataFrame.from_dict(pickup_distance).sort_index().transpose().sort_index().stack().reset_index(drop=True)
    taxi_request_join['pickup_time_in_mins'] = pd.DataFrame.from_dict(pickup_time_in_mins).sort_index().transpose().sort_index().stack().reset_index(drop=True)
 
    #add current timestamp
    taxi_request_join['timestamp'] = timestamp
      
    #drop too far taxi-request pairs
    #taxi_request_join = taxi_request_join[taxi_request_join['pickup_distance']<=max_pickup_dist]
    taxi_request_join = taxi_request_join[taxi_request_join['pickup_time_in_mins']<=max_pickup_time]
    
    #enough energy
    taxi_request_join['enough_range'] = taxi_request_join['end_range'] - taxi_request_join['pickup_distance'] - taxi_request_join['trip_distance']
    taxi_request_join = taxi_request_join[taxi_request_join['enough_range']>0]
    
    #2. prepare data for deep learning dispatch model
    
    used_col = ['end_longitude', 'end_latitude', 'end_range',\
            'origin_longitude', 'origin_latitude', 'trip_distance', 'trip_time',\
            'wait_time', 'pickup_distance', 'pickup_time_in_mins', 'timestamp']
    X = taxi_request_join[used_col]
    
    #use the scaling results of the training samples
    X = sc.transform(X)
    
    #3. use model to make prediction, output probability
    
    y_pred_p = ml_model.predict(X)
    
    #4. consider probability as score, make pickup decision
    
    df = taxi_request_join[['medallion', 'id']]
    df['probability'] = y_pred_p #this probability is pickup probability
    #sort df by df.probability
    df.sort_values(by=['probability'], inplace=True, ascending=False)
    #df.reset_index(drop=True, inplace=True)
    #create solutions dataframe
    v = pd.DataFrame(0, index=taxi_sub1.index, columns=request_sub1.index)
    #give priority to high-probability taxi-request pairs
    while((df.shape[0]>0) and (df.iloc[0]['probability']>=0.5)):
        which_taxi = int(df.iloc[0]['medallion'])
        which_request = int(df.iloc[0]['id'])
        v[which_request][which_taxi] = 1
        df = df[(df['medallion']!=which_taxi) & (df['id']!=which_request)]
    #convert v to dictionary  
    v = v.to_dict(orient="index")
    
    return v
    
#%%
def DeepLearningDispatch2(X, df, taxi_sub_index, request_sub_index):
    #3. use model to make prediction, output probability
    
    y_pred_p = ml_model.predict(X)
    
    #4. consider probability as score, make pickup decision

    df['probability'] = y_pred_p #this probability is pickup probability
    #sort df by df.probability
    df.sort_values(by=['probability'], inplace=True, ascending=False)
    #create solutions dataframe
    v = pd.DataFrame(0, index=taxi_sub_index, columns=request_sub_index)
    #give priority to high-probability taxi-request pairs
    while((df.shape[0]>0) and (df.iloc[0]['probability']>=0.5)):
        which_taxi = int(df.iloc[0]['medallion'])
        which_request = int(df.iloc[0]['id'])
        v[which_request][which_taxi] = 1
        df = df[(df['medallion']!=which_taxi) & (df['id']!=which_request)]
    #convert v to dictionary  
    v = v.to_dict(orient="index")
    
    return v





















