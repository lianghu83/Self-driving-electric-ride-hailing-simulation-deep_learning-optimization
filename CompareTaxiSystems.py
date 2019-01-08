"""
Compare different taxi systems.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
#old taxi system
system_1 = pd.read_csv('H:\\EAV Taxi Data\\'+'10_16_raw'+'\\old_system_output.csv')
#optimization system
system_2 = pd.read_csv('H:\\EAV Taxi Data\\'+'10_16_op'+'\\system_output.csv')
#deep learning system
system_3 = pd.read_csv('H:\\EAV Taxi Data\\'+'10_16_ml'+'\\system_output.csv')

#taxi performance
system_1['occupied_trips'].describe()
system_2['occupied_trips'].describe()
system_3['occupied_trips'].describe()

system_1['travel_distance_total'].describe()
system_2['travel_distance_total'].describe()
system_3['travel_distance_total'].describe()

# =============================================================================
# #plot histograms together
# system_1['travel_distance_total'].max()
# bins = np.linspace(0, 320, 33)
# fig = plt.figure(figsize=(5, 3))
# plt.hist(system_1['travel_distance_total'], bins, alpha=0.5, color='r', label='Current taxis')
# plt.hist(system_3['travel_distance_total'], bins, alpha=0.5, color='g', label='EAV taxis (NN dispatch)')
# plt.legend(loc='upper right')
# plt.xlabel("Travel distance (miles)")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()
# fig.savefig("H:\\EAV Taxi Data\\"+'10_16_ml'+"\\distance"+".jpg", dpi=300)
# =============================================================================

#plot histograms separately
system_1['travel_distance_total'].max()
bins = np.linspace(0, 320, 17)
fig = plt.figure(figsize=(5, 3))
plt.hist([system_1['travel_distance_total'], system_3['travel_distance_total']],
         bins, label=['Current taxis', 'EAV taxis \n(NN dispatch)'], 
         color=['r', 'g'], alpha=0.5)
plt.legend(loc='upper right')
plt.xlabel("Travel distance (miles)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
fig.savefig("H:\\EAV Taxi Data\\"+'10_16_ml'+"\\distance"+".jpg", dpi=300)



system_1['travel_distance_occupied'].describe()
system_2['travel_distance_occupied'].describe()
system_3['travel_distance_occupied'].describe()

system_1['travel_distance_empty'].describe()
system_2['travel_distance_empty'].describe()
system_3['travel_distance_empty'].describe()

#plot empty distance
system_1['travel_distance_empty'].max()
bins = np.linspace(0, 140, 15)
fig = plt.figure(figsize=(5, 3))
plt.hist([system_1['travel_distance_empty'], system_3['travel_distance_empty']],
         bins, label=['Current taxis', 'EAV taxis \n(NN dispatch)'], 
         color=['r', 'g'], alpha=0.5)
plt.legend(loc='upper right')
plt.xlabel("Empty travel distance (miles)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
fig.savefig("H:\\EAV Taxi Data\\"+'10_16_ml'+"\\empty_distance"+".jpg", dpi=300)



system_1['occupied_distance_ratio'] = system_1['travel_distance_occupied'] / system_1['travel_distance_total']
system_1['occupied_distance_ratio'].describe()
system_2['occupied_distance_ratio'] = system_2['travel_distance_occupied'] / system_2['travel_distance_total']
system_2['occupied_distance_ratio'].describe()
system_3['occupied_distance_ratio'] = system_3['travel_distance_occupied'] / system_3['travel_distance_total']
system_3['occupied_distance_ratio'].describe()

#plot
bins = np.linspace(0, 1, 21)
fig = plt.figure(figsize=(5, 3))
plt.hist([system_1['occupied_distance_ratio'], system_3['occupied_distance_ratio']],
         bins, label=['Current taxis', 'EAV taxis \n(NN dispatch)'], 
         color=['r', 'g'], alpha=0.5)
plt.legend(loc='upper left')
plt.xlabel("Occupancy")
plt.ylabel("Frequency")
plt.xticks(np.linspace(0, 1, 11))
plt.tight_layout()
plt.show()
fig.savefig("H:\\EAV Taxi Data\\"+'10_16_ml'+"\\occupancy"+".jpg", dpi=300)



sum(system_1['travel_distance_total'])
sum(system_2['travel_distance_total'])
sum(system_3['travel_distance_total'])

system_2['go_charging_time'].describe() 
system_3['go_charging_time'].describe() 

plt.hist(system_2['go_charging_time'])

system_2['charging_time'].describe() 
system_3['charging_time'].describe() 

plt.hist(system_2['charging_time'])

system_2['total_time_for_charging'] = system_2['go_charging_time'] + system_2['charging_time']
system_2['total_time_for_charging'].describe() 
system_3['total_time_for_charging'] = system_3['go_charging_time'] + system_3['charging_time']
system_3['total_time_for_charging'].describe() 

plt.hist(system_2['total_time_for_charging'])

system_2['charging_number'].describe() 
system_2['charging_number'].value_counts(normalize=True)
system_3['charging_number'].describe() 
system_3['charging_number'].value_counts(normalize=True)

#%%
#use excel to plot activity changes
status_change_2 = pd.read_excel('H:\\EAV Taxi Data\\'+'10_16_op'+'\\status_change.xlsx')
status_change_2['serving'] = status_change_2['called'] + status_change_2['occupied']
status_change_2['charging'] = status_change_2['go charging'] + status_change_2['start charging']

r = status_change_2['minute']
bar1 = status_change_2['waiting']
bar2 = status_change_2['serving']
bar3 = status_change_2['charging']

plt.bar(r, bar1, color='#7f6d5f')
plt.bar(r, bar2, bottom=bar1, color='#557f2d')
plt.bar(r, bar3, bottom=bar1+bar2, color='#2d7f5e')



#%%computation time
op_times = pd.read_csv("H:\\EAV Taxi Data\\10_16_op\\optimization_times.csv")
ml_times = pd.read_csv("H:\\EAV Taxi Data\\10_16_ml\\deep_learning_times.csv")

op_times['0'] *= 60
ml_times['0'] *= 60

op_times.describe()
ml_times.describe()

#plot histograms
bins = np.linspace(0, 120, 13)
fig = plt.figure(figsize=(5, 3))
plt.hist([op_times['0'], ml_times['0']],
         bins,
         label=['Optimization dispatch', 'Neural network dispatch'], 
         color=['b', 'g'], alpha=0.5)
plt.legend(loc='upper right')
plt.xlabel("Model solving time (sec)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
fig.savefig("H:\\EAV Taxi Data\\"+'10_16_ml'+"\\solving_time"+".jpg", dpi=300)




