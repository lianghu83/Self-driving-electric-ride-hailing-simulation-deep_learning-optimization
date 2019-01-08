"""
This file measures the delay/performance of requests
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Parameters import start_time, max_wait_time, max_pickup_time

scenario = '10_16_ml'

### requests' delay ###
request = pd.read_csv("H:\\EAV Taxi Data\\"+scenario+"\\request_output.csv")
request.dtypes
num_request = request.shape[0]

#percent of completed requests
request['served'].describe()
request[request['served']==False].shape[0]/request.shape[0]
request[request['served']==True].shape[0]/request.shape[0]

#delete False
request_rej = request[request['served']==False]
request = request[request['served']==True]

#delay time
request['wait_time'].describe()
np.percentile(request['wait_time'], 95)

# =============================================================================
# bins = np.linspace(0, 45, 46)
# fig = plt.figure()
# plt.hist(request['wait_time'], bins, color='y')
# plt.xlabel("Delay time (minutes)")
# plt.ylabel("Frequency")
# plt.xticks(np.linspace(0, 45, 10))
# fig.savefig("H:\\EAV Taxi Data\\"+scenario+"\\delay_time"+".jpg", figsize=(5, 3), dpi=300)
# plt.show()
# =============================================================================

#taxi pickup time
request['taxi_pickup_time'].describe()
np.percentile(request['taxi_pickup_time'], 90)
#% that can be picked up within x minutes
[request[request['taxi_pickup_time']<=x].shape[0] / request.shape[0] for x in [1,3,5,7,10,15]]

bins = np.linspace(0, 30, 31)
fig = plt.figure(figsize=(5, 3))
plt.hist(request['taxi_pickup_time'], bins, color='g', alpha=0.5)
plt.xlabel("Taxi pickup time (min)")
plt.ylabel("Frequency")
plt.yticks(np.linspace(0, 8000, 9))
plt.tight_layout()
plt.show()
#fig.savefig("H:\\EAV Taxi Data\\"+scenario+"\\pickup_time"+".jpg",  dpi=300)


#wait time before accepted
request['wait_time_before_accepted'] = request['wait_time']-request['taxi_pickup_time']
request['wait_time_before_accepted'].describe()
np.percentile(request['wait_time_before_accepted'], 95)

# =============================================================================
# bins = np.linspace(0, 15, 16)
# fig = plt.figure()
# plt.hist(request['wait_time_before_accepted'], bins, color='C1')
# plt.xlabel("Wait time until accepted (minutes)")
# plt.ylabel("Frequency")
# plt.xticks(np.linspace(0, 15, 16))
# fig.savefig("H:\\EAV Taxi Data\\"+scenario+"\\wait_time_accepted"+".jpg", figsize=(5, 3), dpi=300)
# plt.show()
# =============================================================================

def CompareTaxiPickupTime():
    op = pd.read_csv("H:\\EAV Taxi Data\\"+'10_16_op'+"\\request_output.csv")
    op = op[op['served']==True]
    op = [op[op['taxi_pickup_time']<=x].shape[0] / op.shape[0] for x in range(1,31)]
    ml = pd.read_csv("H:\\EAV Taxi Data\\"+'10_16_ml'+"\\request_output.csv")
    ml = ml[ml['served']==True]
    ml = [ml[ml['taxi_pickup_time']<=x].shape[0] / ml.shape[0] for x in range(1,31)]
    
    fig = plt.figure(figsize=(5, 3))
    plt.plot(op, color='b', alpha=0.5, label='EAV taxis \n(optimization dispatch)', linestyle='dashed')
    plt.plot(ml, color='g', alpha=0.5, label='EAV taxis \n(NN dispatch)', linestyle='dashed')
    plt.legend(loc='lower right')
    plt.xlabel("Taxi pickup time (min)")
    plt.ylabel("Prob. of occurrence")
    x_tick = ['\u2264'+'1', '', '\u2264'+'3', '','\u2264'+'5',
              '', '\u2264'+'7', '', '', '\u2264'+'10',
              '','','','','\u2264'+'15',
              '','','','','\u2264'+'20',
              '','','','','\u2264'+'25',
              '','','','','\u2264'+'30']
    plt.xticks(range(0,30), x_tick)
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.tight_layout()
    plt.show()
    fig.savefig("H:\\EAV Taxi Data\\"+'10_16_ml'+"\\compare_pickup"+".jpg", dpi=300)
    
CompareTaxiPickupTime()   
    
    
    
def CompareAcceptOrRejTime():
    op = pd.read_csv("H:\\EAV Taxi Data\\"+'10_16_op'+"\\request_output.csv")
    op = op[op['served']==True]
    op['wait_time_before_accepted'] = op['wait_time']-op['taxi_pickup_time']
    op = [op[op['wait_time_before_accepted']<=x].shape[0] / num_request for x in range(0,16)]
    ml = pd.read_csv("H:\\EAV Taxi Data\\"+'10_16_ml'+"\\request_output.csv")
    ml = ml[ml['served']==True]
    ml['wait_time_before_accepted'] = ml['wait_time']-ml['taxi_pickup_time']
    ml = [ml[ml['wait_time_before_accepted']<=x].shape[0] / num_request for x in range(0,16)]
    
    fig = plt.figure(figsize=(5, 3))
    plt.plot(op, color='b', alpha=0.5, label='EAV taxis \n(optimization dispatch)', linestyle='dashed')
    plt.plot(ml, color='g', alpha=0.5, label='EAV taxis \n(NN dispatch)', linestyle='dashed')
    plt.legend(loc='lower right')
    plt.xlabel("Waiting time before accepted (min)")
    plt.ylabel("Prob. of occurrence")
    x_tick = ['0', '\u2264'+'1', '', '\u2264'+'3', '','\u2264'+'5',
              '', '\u2264'+'7', '', '', '\u2264'+'10',
              '','','','','\u2264'+'15']
    plt.xticks(range(0,16), x_tick)
    plt.yticks([0.90,0.92,0.94,0.96,0.98,1])
    plt.tight_layout()
    plt.show()
    fig.savefig("H:\\EAV Taxi Data\\"+'10_16_ml'+"\\compare_accept_rej"+".jpg", dpi=300)
    
CompareAcceptOrRejTime()       


#%%for all request

#% that can be accepted 0-x+1 min
[request[request['wait_time_before_accepted']<=x].shape[0] / num_request for x in [0,1,2,3,4]]



#wait time before accepted or rejected
combine_wait = request['wait_time_before_accepted'].append(request_rej['wait_time'])
combine_wait[combine_wait==16] = 15
combine_wait.describe()
np.percentile(combine_wait, 90)

# =============================================================================
# bins = np.linspace(0, 15, 16)
# fig = plt.figure()
# plt.hist(combine_wait, bins, color='C1')
# plt.xlabel("Wait time until accepted/rejected (minutes)")
# plt.ylabel("Frequency")
# plt.xticks(np.linspace(0, 15, 16))
# fig.savefig("H:\\EAV Taxi Data\\"+scenario+"\\wait_time_accepted_rejected"+".jpg", figsize=(5, 3), dpi=300)
# plt.show()
# =============================================================================



#total cost of system
temp = request_rej['wait_time']
temp[temp==16] = 15
sum(request['taxi_pickup_time']) + (sum(temp) + max_pickup_time*len(temp))
sum(request['wait_time']) + (sum(temp) + max_pickup_time*len(temp))



#plot GPS of rejected requests
fig = plt.figure(figsize=(5, 3))
plt.scatter(x=request_rej['origin_longitude'], y=request_rej['origin_latitude'], color='r')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
fig.savefig("H:\\EAV Taxi Data\\"+scenario+"\\GPS_rejected"+".jpg", dpi=300)



#%%for all rejected requestes
#convert origin_timestamp into hours
request_rej['origin_hour'] = (request_rej['origin_timestamp']-start_time)/3600
#plot time distribution histogram
plt.hist(request_rej['origin_hour'], bins=72)



#%%for taxis at the end of simulation
### taxi SOC ###
#taxi end SOC
taxi = pd.read_csv("H:\\EAV Taxi Data\\"+scenario+"\\taxi_output.csv")
taxi.dtypes

#summary
taxi['end_range'].describe()
np.percentile(taxi['end_range'], 90)
np.percentile(taxi['end_range'], 95)

#histogram
plt.hist(taxi['end_range'])
plt.xlabel("Remaining range (mi)")
plt.ylabel("Frequency")

#boxplot
plt.boxplot(taxi['end_range'])

#the end location of taxis
plt.scatter(x=taxi['end_longitude'], y=taxi['end_latitude'])
plt.scatter(x=request_rej['origin_longitude'], y=request_rej['origin_latitude'])





















