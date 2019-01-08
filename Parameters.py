"""
Set parameters
"""

from geopy.distance import great_circle
#import numpy as np

scenario = '10_16_ml_0.95' #set up senario

EV_range = 200.0 #miles

speed_average_NYC = 13.0 #mph

electricity_consumption_rate = 0.3 #kWh/mi

#charge_to = 1 #charge to 100%

need_charge = 0.1 #SOC<10% range

charge_to = 0.8*EV_range #may stop charging when charging to 80% of range

low_range = 20.0 #need charge, miles

time_interval = 60 #seconds; 1 min

pickup_ub = 60 #maximum pickup time, min, performs as penalty

max_pickup_time = 30.0 #maximum pickup time, minutes
max_pickup_dist = max_pickup_time*speed_average_NYC/60 #maximum pickup disntance, miles
grid_size = max_pickup_dist/1.4142

max_wait_time = 15.0

charging_power = 50.0 #kW

charge_minimum_time = 30 #charge for at least 30 minutes

#create time interval list
#1 day
start_time = 1381896000 #2013-10-16 00:00:00
end_time = 1381982400 #2013-10-17 00:00:00
#1 week
#start_time = 1378612800 #2013-09-08 00:00:00
#end_time = 1379217600 #2013-09-15 00:00:00
#1 day
#start_time = 1382500800 
#end_time = 1382587200 
#3 days
#start_time = 1378612800 #2013-09-08 00:00:00
#end_time = 1378872000 #2013-09-11 00:00:00
#3 days
#start_time = 1378785600 #2013-09-10 00:00:00
#end_time = start_time+86400*3 #2013-09-13 00:00:00
time_range = range(start_time, end_time, time_interval)

#separate NYC to grid
up_left = (41.1,-74.25) #latitude first
bottom_right = (40.4, -73.5)

long_dist = great_circle(up_left, (up_left[0],bottom_right[1])).miles
long_step = abs(up_left[1]-bottom_right[1])/(long_dist/grid_size)
#long_cord = np.arange(up_left[1], bottom_right[1], long_step)
#long_cord = np.append(long_cord, bottom_right[1])
#len(long_cord)

lat_dist = great_circle(up_left, (bottom_right[0],up_left[1])).miles
lat_step = abs(up_left[0]-bottom_right[0])/(lat_dist/grid_size)
#lat_cord = np.arange(up_left[0], bottom_right[0], -lat_step)
#lat_cord = np.append(lat_cord, bottom_right[0])
#len(lat_cord)