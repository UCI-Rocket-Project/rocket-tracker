import numpy as np
from dataclasses import dataclass
@dataclass
class GroundTruthTrackingData:
    '''
    Dataclass for tracking data from the ground truth simulation
    '''
    pixel_coordinates: tuple[int,int]
    enu_coordinates: np.ndarray
    az_alt: tuple[float,float]

@dataclass
class TelemetryData:
    '''
    Dataclass for telemetry data from the rocket. Right now it excludes the IMU data (accel and gyro)
    '''
    gps_lat: float
    gps_lng: float
    altimeter_reading: float

def gps_to_enu(gps_coord: np.ndarray, origin_gps: np.ndarray) -> np.ndarray:
    '''
    Converts GPS coordinates to ENU (East, North, Up) coordinates
    '''
    lat,lng,alt = gps_coord
    lat0,lng0,alt0 = origin_gps
    R = 6_371_000 # radius of earth in meters
    x = (lng-lng0)*R*np.cos(lat0)
    y = (lat-lat0)*R
    z = alt-alt0
    return np.array([x,y,z])

def enu_to_gps(enu_coord: np.ndarray, origin_gps: np.ndarray) -> np.ndarray:
    '''
    Converts ENU (East, North, Up) coordinates to GPS coordinates
    '''
    x,y,z = enu_coord
    lat0,lng0,alt0 = origin_gps
    R = 6_371_000 # radius of earth in meters
    lat = lat0+y/R
    lng = lng0+x/(R*np.cos(lat0))
    alt = alt0+z
    return np.array([lat,lng,alt])

