"""
Classes
"""

from Parameters import EV_range, start_time, low_range, charging_power, electricity_consumption_rate
#import pandas as pd
import random



class TaxiObj(object):
    
    #system_timestamp = start_time
    taxi_range = EV_range
    
    def __init__(self, medallion, long, lat):
        #taxi start range
        temp_range = random.uniform(low_range, EV_range)     
        #taxi dictionary for current status
        self.dict = {'medallion': medallion, #taxi medallion
                     'status': 'waiting', #taxi status
                     'start_timestamp': start_time, #current time
                     'start_longitude': long, #taxi current GPS longitude
                     'start_latitude': lat, #taxi current GPS latitude
                     'start_range': temp_range, #taxi start range
                     'status_distance': 0.0, #travel distance for this status
                     'status_time': -1, #travel time for this status
                     'end_timestamp': -1, #taxi end timestamp
                     'end_longitude': long, #taxi end GPS longitude
                     'end_latitude': lat, #taxi end GPS latitude
                     'end_range': temp_range} #taxi end range        
        #store taxi activities
        self.activities = [self.dict]
        
    #taxi wait at somewhere
    def startWaiting(self, timestamp):
        self.dict['status'] = 'waiting'
        self.dict['start_timestamp'] = timestamp
        self.dict['start_longitude'] = self.dict['end_longitude']
        self.dict['start_latitude'] = self.dict['end_latitude']
        self.dict['start_range'] = self.dict['end_range']
        self.dict['status_distance'] = 0.0
        self.dict['status_time'] = -1
        self.dict['end_timestamp'] = -1
        self.dict['end_longitude'] = self.dict['start_longitude']
        self.dict['end_latitude'] = self.dict['start_latitude']
        self.dict['end_range'] = self.dict['start_range']
        self.activities.append(self.dict)
        
    #taxi gets called to pickup customer
    def getCalled(self, request, timestamp, taxi_pickup_dist, taxi_pickup_time): #request is an instance
        self.dict['status'] = 'called'
        self.dict['start_timestamp'] = timestamp
        self.dict['start_longitude'] = self.dict['end_longitude']
        self.dict['start_latitude'] = self.dict['end_latitude']
        self.dict['start_range'] = self.dict['end_range']
        self.dict['status_distance'] = taxi_pickup_dist
        self.dict['status_time'] = taxi_pickup_time
        self.dict['end_timestamp'] = self.dict['start_timestamp'] + taxi_pickup_time*60
        self.dict['end_longitude'] = request.origin_longitude
        self.dict['end_latitude'] = request.origin_latitude
        self.dict['end_range'] = self.dict['start_range'] - taxi_pickup_dist
        self.activities.append(self.dict)
    
    #taxi serves customer
    def serveCustomer(self, request, timestamp):
        self.dict['status'] = 'occupied'
        self.dict['start_timestamp'] = timestamp
        self.dict['start_longitude'] = self.dict['end_longitude']
        self.dict['start_latitude'] = self.dict['end_latitude']
        self.dict['start_range'] = self.dict['end_range']
        self.dict['status_distance'] = request.trip_distance
        self.dict['status_time'] = request.trip_time
        self.dict['end_timestamp'] = self.dict['start_timestamp'] + self.dict['status_time']*60
        self.dict['end_longitude'] = request.destination_longitude
        self.dict['end_latitude'] = request.destination_latitude
        self.dict['end_range'] = self.dict['start_range'] - self.dict['status_distance']
        self.activities.append(self.dict)
              
    #taxi goes to charging station
    def goToCharging(self, request, timestamp): #NEED MODIFICATION HERE
        self.dict['status'] = 'go charging'
        self.dict['start_timestamp'] = timestamp
        self.dict['start_longitude'] = self.dict['end_longitude']
        self.dict['start_latitude'] = self.dict['end_latitude']
        self.dict['start_range'] = self.dict['end_range']
        self.dict['status_distance'] = request.cs_distance
        self.dict['status_time'] = request.cs_travel_time
        self.dict['end_timestamp'] = self.dict['start_timestamp'] + self.dict['status_time']*60
        self.dict['end_longitude'] = request.cs_longitude
        self.dict['end_latitude'] = request.cs_latitude
        self.dict['end_range'] = self.dict['start_range'] - self.dict['status_distance']
        self.activities.append(self.dict)
        
    #taxi starts charging
    def startCharging(self, timestamp): #NEED MODIFICATION HERE
        self.dict['status'] = 'start charging'
        self.dict['start_timestamp'] = timestamp
        self.dict['start_longitude'] = self.dict['end_longitude']
        self.dict['start_latitude'] = self.dict['end_latitude']
        self.dict['start_range'] = self.dict['end_range']
        self.dict['status_distance'] = 0.0
        #charging_time = int((self.range-self.start_range)*electricity_consumption_rate/charging_power*60) #min
        self.dict['status_time'] = -1
        self.dict['end_timestamp'] = -1 #self.start_timestamp + self.status_time*60
        self.dict['end_longitude'] = self.dict['start_longitude']
        self.dict['end_latitude'] = self.dict['start_latitude']
        self.dict['end_range'] = self.dict['start_range'] #update range in endCharging()
        self.activities.append(self.dict)
        
    #taxi ends charging
    def endCharging(self, timestamp):
        self.dict['status'] = 'waiting'
        add_range = min((timestamp-self.dict['start_timestamp'])/3600*charging_power/electricity_consumption_rate, self.taxi_range-self.dict['start_range'])
        self.dict['start_timestamp'] = timestamp
        self.dict['start_longitude'] = self.dict['end_longitude']
        self.dict['start_latitude'] = self.dict['end_latitude']
        self.dict['start_range'] = self.dict['start_range'] + add_range
        self.dict['status_distance'] = 0.0
        self.dict['status_time'] = -1
        self.dict['end_timestamp'] = -1
        self.dict['end_longitude'] = self.dict['start_longitude']
        self.dict['end_latitude'] = self.dict['start_latitude']
        self.dict['end_range'] = self.dict['start_range']
        self.activities.append(self.dict)
        
    #taxi relocates
    def relocate(self, destination, timestamp):
        pass
    
    #find the nearest charging station
    def findChargingStation(self, timestamp):
        pass
    
    def __repr__(self):
        return self.activities[-1]
        
        
        
class RequestObj(object):
    
    #system_timestamp = start_time
    
    def __init__(self, row): #row is a request pd.Series
        #request id
        self.id = row['id']

        #request time
        self.origin_timestamp = row['pickup_datetime']
        #request origin GPS longitude
        self.origin_longitude = row['pickup_longitude']
        #request origin GPS latitude
        self.origin_latitude = row['pickup_latitude']
        
        #trip distance
        self.trip_distance = row['trip_distance']
        #trip travel time
        self.trip_time = row['trip_time_in_mins'] #minutes
        
        #arrival time that should be, without any delay
        self.destination_timestamp = row['dropoff_datetime']
        #request destination GPS longitude
        self.destination_longitude = row['dropoff_longitude']
        #request destination GPS latitude
        self.destination_latitude = row['dropoff_latitude']
        
        #nearest charging station id
        #self.cs_id = row['CS_id']
        #distance to the charging station
        self.cs_distance = row['CS_distance']
        #travel time to the charging station
        self.cs_travel_time = row['CS_travel_time_in_mins'] #minutes
        #charging station longitude
        self.cs_longitude = row['CS_longitude']
        #charging station latitude
        self.cs_latitude = row['CS_latitude']
      
        #time from request to taxi arrives
        self.wait_time = 0
        #request served or not
        self.served = False
        #served by which taxi
        self.served_taxi = None
        #time for the called taxi coming
        self.taxi_pickup_time = None
        #picked up time by a taxi in fact
        self.pickup_timestamp = None
        #dropoff time in fact
        self.dropoff_timestamp = None
        
    #request get rejected at a time interval
    def getRejected(self):
        self.wait_time += 1
        
    #request get accepted, a taxi comes
    def getAccepted(self, medallion, taxi_pickup_time):
        self.taxi_pickup_time = taxi_pickup_time
        self.wait_time += taxi_pickup_time
        self.served = True
        self.served_taxi = medallion
     
    #request is served by taxi
    def getServed(self, start_timestamp):
        self.pickup_timestamp = start_timestamp
        self.dropoff_timestamp = self.pickup_timestamp + self.trip_time*60
        
        
        
"""            
class ChargingStation(object):
    
    def __init__(self):
        #station id
        self.id = None
        #GPS longitude
        self.longitude = None
        #GPS latitude
        self.latitude = None
        #number of chargers
        self.chargers = 1
        #charger power
        self.power = charging_power
        #which taxi is using
        self.taxi = []
        #taxi start charging timestamp
        #self.start_charging_timestamp = None
        #taxi end charging timestamp
        #self.end_charging_timestamp = None
        #how many cars are using this station
        self.taxi_amount = 0
        #keep charging records
        self.charging_events = {}
    
    #a taxi comes and starts charging    
    def startService(self, taxi): #taxi is an instance
        self.taxi.append(taxi.medallion)
        self.taxi_amount = len(self.taxi)

    #a taxi leaves and ends charging
    def endService(self, taxi):
        self.taxi.remove(taxi.medallion)
        self.taxi_amount = len(self.taxi)
        
        
        
#define depot or not?
class Depot(object):
    pass
"""        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        