"""
Compare different taxi systems.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as stats

#%%
#old taxi system
system_1 = pd.read_csv('C:\\EAV Taxi Data\\'+'10_16_raw'+'\\old_system_output.csv')
#optimization system
system_2 = pd.read_csv('C:\\EAV Taxi Data\\'+'10_16_op'+'\\system_output.csv')
#deep learning system
system_3 = pd.read_csv('C:\\EAV Taxi Data\\'+'10_16_ml_R1'+'\\system_output.csv')

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
         bins, label=['Current taxis', 'EAV taxis'], #'EAV taxis \n(NN dispatch)'
         color=['r', 'g'], alpha=0.5)
plt.legend(loc='upper right')
plt.xlabel("Travel distance (miles)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
fig.savefig("C:\\EAV Taxi Data\\"+'10_16_ml_R1'+"\\distance"+".jpg", dpi=300)



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
         bins, label=['Current taxis', 'EAV taxis'], 
         color=['r', 'g'], alpha=0.5)
plt.legend(loc='upper right')
plt.xlabel("Empty travel distance (miles)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
fig.savefig("C:\\EAV Taxi Data\\"+'10_16_ml_R1'+"\\empty_distance"+".jpg", dpi=300)



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
         bins, label=['Current taxis', 'EAV taxis'], 
         color=['r', 'g'], alpha=0.5)
plt.legend(loc='upper left')
plt.xlabel("Occupancy")
plt.ylabel("Frequency")
plt.xticks(np.linspace(0, 1, 11))
plt.tight_layout()
plt.show()
fig.savefig("C:\\EAV Taxi Data\\"+'10_16_ml_R1'+"\\occupancy"+".jpg", dpi=300)



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
status_change_2 = pd.read_excel('C:\\EAV Taxi Data\\'+'10_16_op'+'\\status_change.xlsx')
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
op_times = pd.read_csv("C:\\EAV Taxi Data\\10_16_op\\optimization_times.csv")
ml_times = pd.read_csv("C:\\EAV Taxi Data\\10_16_ml_R1\\deep_learning_times.csv")

op_times['0'] *= 60
ml_times['0'] *= 60

op_times.describe()
ml_times.describe()

ml_times[ml_times['0']<=15].count()/ml_times.shape[0]

#plot histograms
bins = np.linspace(0, 120, 13)
fig = plt.figure(figsize=(5, 3))
plt.hist([op_times['0'], ml_times['0']],
         bins,
         label=['Optimization model', 'ANN model'], 
         color=['b', 'g'], alpha=0.5)
plt.legend(loc='upper right')
plt.xlabel("Computation time (sec)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
fig.savefig("C:\\EAV Taxi Data\\"+'10_16_ml_R1'+"\\computation_time"+".jpg", dpi=300)



#%%computation time using different sample size
z_critical = stats.norm.ppf(q = 0.975)

#prepare data for optimization
op_sample_means = []
op_intervals = []

op_1 = pd.read_csv("C:\\EAV Taxi Data\\10_16_op_1%\\optimization_times.csv")
op_3 = pd.read_csv("C:\\EAV Taxi Data\\10_16_op_3%\\optimization_times.csv")
op_5 = pd.read_csv("C:\\EAV Taxi Data\\10_16_op\\optimization_times.csv")
op_7 = pd.read_csv("C:\\EAV Taxi Data\\10_16_op_7%\\optimization_times.csv")
op_10 = pd.read_csv("C:\\EAV Taxi Data\\10_16_op_10%\\optimization_times.csv")

op_sample_means.append(np.mean(op_1)[0]*60)
op_sample_means.append(np.mean(op_3)[0]*60)
op_sample_means.append(np.mean(op_5)[0]*60)
op_sample_means.append(np.mean(op_7)[0]*60)
op_sample_means.append(np.mean(op_10)[0]*60)

margin_of_error = z_critical * (op_1.std()[0]/math.sqrt(len(op_1)))*60
op_intervals.append((op_sample_means[0] - margin_of_error, op_sample_means[0] + margin_of_error))

margin_of_error = z_critical * (op_3.std()[0]/math.sqrt(len(op_3)))*60
op_intervals.append((op_sample_means[1] - margin_of_error, op_sample_means[1] + margin_of_error))

margin_of_error = z_critical * (op_5.std()[0]/math.sqrt(len(op_5)))*60
op_intervals.append((op_sample_means[2] - margin_of_error, op_sample_means[2] + margin_of_error))

margin_of_error = z_critical * (op_7.std()[0]/math.sqrt(len(op_7)))*60
op_intervals.append((op_sample_means[3] - margin_of_error, op_sample_means[3] + margin_of_error))

margin_of_error = z_critical * (op_10.std()[0]/math.sqrt(len(op_10)))*60
op_intervals.append((op_sample_means[4] - margin_of_error, op_sample_means[4] + margin_of_error))


#prepare data for NN
ml_sample_means = []
ml_intervals = []

ml_1 = pd.read_csv("C:\\EAV Taxi Data\\10_16_ml_R1_1%\\deep_learning_times.csv")
ml_3 = pd.read_csv("C:\\EAV Taxi Data\\10_16_ml_R1_3%\\deep_learning_times.csv")
ml_5 = pd.read_csv("C:\\EAV Taxi Data\\10_16_ml_R1\\deep_learning_times.csv")
ml_7 = pd.read_csv("C:\\EAV Taxi Data\\10_16_ml_R1_7%\\deep_learning_times.csv")
ml_10 = pd.read_csv("C:\\EAV Taxi Data\\10_16_ml_R1_10%\\deep_learning_times.csv")

ml_sample_means.append(np.mean(ml_1)[0]*60)
ml_sample_means.append(np.mean(ml_3)[0]*60)
ml_sample_means.append(np.mean(ml_5)[0]*60)
ml_sample_means.append(np.mean(ml_7)[0]*60)
ml_sample_means.append(np.mean(ml_10)[0]*60)

margin_of_error = z_critical * (ml_1.std()[0]/math.sqrt(len(ml_1)))*60
ml_intervals.append((ml_sample_means[0] - margin_of_error, ml_sample_means[0] + margin_of_error))

margin_of_error = z_critical * (ml_3.std()[0]/math.sqrt(len(ml_3)))*60
ml_intervals.append((ml_sample_means[1] - margin_of_error, ml_sample_means[1] + margin_of_error))

margin_of_error = z_critical * (ml_5.std()[0]/math.sqrt(len(ml_5)))*60
ml_intervals.append((ml_sample_means[2] - margin_of_error, ml_sample_means[2] + margin_of_error))

margin_of_error = z_critical * (ml_7.std()[0]/math.sqrt(len(ml_7)))*60
ml_intervals.append((ml_sample_means[3] - margin_of_error, ml_sample_means[3] + margin_of_error))

margin_of_error = z_critical * (ml_10.std()[0]/math.sqrt(len(ml_10)))*60
ml_intervals.append((ml_sample_means[4] - margin_of_error, ml_sample_means[4] + margin_of_error))


#plot
fig = plt.figure(figsize=(5, 3))
plt.errorbar(x=['1%','3%','5%','7%','10%'], 
             y=op_sample_means, 
             yerr=[(top-bot)/2 for top,bot in op_intervals],
             fmt='-bo',
             alpha=0.5)
plt.errorbar(x=['1%','3%','5%','7%','10%'], 
             y=ml_sample_means, 
             yerr=[(top-bot)/2 for top,bot in ml_intervals],
             fmt='-go',
             alpha=0.5)
plt.legend(['Optimization model', 'ANN model'], loc='upper left')
plt.xlabel("Sample size")
plt.ylabel("Computation time (sec)")
plt.tight_layout()
plt.show()
fig.savefig("C:\\EAV Taxi Data\\"+'10_16_ml_R1'+"\\computation_time_sample_size"+".jpg", dpi=300)



#%% Compare objective values
obj_values_op = pd.read_csv("C:\\EAV Taxi Data\\10_16_obj\\obj_values_op.csv")
obj_values_ml = pd.read_csv("C:\\EAV Taxi Data\\10_16_obj\\obj_values_ml.csv")

obj_values_diff = obj_values_ml - obj_values_op
obj_values_rel = obj_values_diff / obj_values_op

[obj_values_rel[obj_values_rel['0']>=x].shape[0] / len(obj_values_rel) for x in [0,-0.1,-0.2,-0.3,-0.4,-0.5]]
(0.72720-0.26437)/(1-0.26437)
(0.94810-0.26437)/(1-0.26437)

# plot
fig = plt.figure(figsize=(5, 3))
plt.hist(obj_values_rel['0'],
         color='grey',
         #alpha=1,
         bins=np.linspace(-0.5, 0, num=11))
plt.xlabel("Reduction in objective values")
plt.ylabel("Frequency")
plt.xlim([-0.5, 0])
plt.xticks(np.linspace(-0.5, 0, num=11),
           ['-50%','','-40%','','-30%','','-20%','','-10%','','0'])
#plt.xticks(['-50%','-45%','-40%','-35%','-30%','-25%','-20%','-15%','-10%','-5%','0%'])
plt.tight_layout()
plt.show()
fig.savefig("C:\\EAV Taxi Data\\"+'10_16_obj'+"\\compare_obj_value"+".jpg", dpi=300)

















