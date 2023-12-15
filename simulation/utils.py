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
    accel_x: float
    accel_y: float
    accel_z: float