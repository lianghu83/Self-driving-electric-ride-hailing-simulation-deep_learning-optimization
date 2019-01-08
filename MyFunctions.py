"""
Functions
"""

from Parameters import *
from geopy.distance import great_circle



def FindGrid(GPS, long_step, lat_step, up_left, bottom_right): #GPS=(lat, long)
    long_num = int(abs(GPS[1]-up_left[1])/long_step)
    lat_num = int(abs(GPS[0]-up_left[0])/lat_step)
    #grid numbering starts from (0,0)
    return (long_num,lat_num)



def EstimateActualDistance(GPS_1, GPS_2):
    #GPS=(lat, long)
    return 1.4413*great_circle(GPS_1, GPS_2).miles + 0.1383 #miles
    