from .flight_model import test_flight
from src.utils import TelemetryData
import numpy as np
import pymap3d as pm

class Rocket:
    def __init__(self, pad_geodetic_pos: np.ndarray, launch_time: float = 0):
        '''
        pad_geodetic_pos: np.ndarray of length 3, [lat, lon, alt]
        '''
        self.pad_geodetic_pos = pad_geodetic_pos
        self.initial_alt = test_flight.z(0)
        self.launch_time = launch_time

    def get_position(self, time):
        '''
        Return xyz position in ENU frame centered on the launch pad
        '''
        if time < self.launch_time:
            time = 0
        else:
            time -= self.launch_time
        return np.array([test_flight.x(time), test_flight.y(time), test_flight.z(time)])

    def get_position_ecef(self, time):
        xyz_enu = self.get_position(time)
        return np.array(pm.enu2ecef(*xyz_enu, *self.pad_geodetic_pos))

    def get_velocity(self, time):
        if time < self.launch_time:
            time = 0
        else:
            time -= self.launch_time
        return np.array([test_flight.vx(time), test_flight.vy(time), test_flight.vz(time)])
    
    def get_acceleration(self, time):
        if time < self.launch_time:
            time = 0
        else:
            time -= self.launch_time
        return np.array([test_flight.ax(time), test_flight.ay(time), test_flight.az(time)])

    def get_telemetry(self, time):
        GPS_NOISE_STD = 0#1e-4
        ALT_NOISE_STD = 0#10
        ACC_NOISE_STD = 0#1e-2

        xyz = self.get_position(time)

        gps_pos = pm.enu2geodetic(*xyz, *self.initial_position)

        if time < self.launch_time:
            time = 0
        else:
            time -= self.launch_time
        return TelemetryData(
            gps_lat=test_flight.latitude(time) + np.random.normal(GPS_NOISE_STD),
            gps_lng=test_flight.longitude(time) + np.random.normal(GPS_NOISE_STD),
            altimeter_reading=test_flight.z(time) + np.random.normal(ALT_NOISE_STD),
            accel_x = test_flight.ax(time) + np.random.normal(ACC_NOISE_STD),
            accel_y = test_flight.ay(time) + np.random.normal(ACC_NOISE_STD),
            accel_z = test_flight.az(time) + np.random.normal(ACC_NOISE_STD),
            time = time
        )
