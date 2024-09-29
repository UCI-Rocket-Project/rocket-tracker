from .flight_model import test_flight
from src.utils import TelemetryData
import numpy as np
import pymap3d as pm
from scipy.spatial.transform import Rotation as R

class Rocket:
    def __init__(self, pad_geodetic_pos: np.ndarray, launch_time: float = 0):
        '''
        pad_geodetic_pos: np.ndarray of length 3, [lat, lon, alt]
        '''
        self.pad_geodetic_pos = pad_geodetic_pos
        self.initial_alt = test_flight.z(0) - pad_geodetic_pos[2]
        self.launch_time = launch_time

    def get_position(self, time):
        '''
        Return xyz position in ENU frame centered on the launch pad
        '''
        if time < self.launch_time:
            time = 0
        else:
            time -= self.launch_time
        return np.array([test_flight.x(time), test_flight.y(time), test_flight.z(time) - self.pad_geodetic_pos[2] - self.initial_alt])

    def get_position_ecef(self, time):
        xyz_enu = self.get_position(time)
        return np.array(pm.enu2ecef(*xyz_enu, *self.pad_geodetic_pos))

    def get_velocity(self, time):
        '''
        Returns xyz velocity in ENU frame centered on the launch pad
        '''
        if time < self.launch_time:
            time = 0
        else:
            time -= self.launch_time
        return np.array([test_flight.vx(time), test_flight.vy(time), test_flight.vz(time)])
    
    def get_velocity_ecef(self, time):
        vel_enu = self.get_velocity(time)
        current_pos_geodetic = pm.ecef2geodetic(self.get_position_ecef(time))
        return np.array(pm.enu2ecef(*vel_enu, *current_pos_geodetic))
    
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

        adjusted_time = max(0,time - self.launch_time)

        v_enu = self.get_velocity(adjusted_time)

        return TelemetryData(
            gps_lat=test_flight.latitude(adjusted_time) + np.random.normal(0,GPS_NOISE_STD),
            gps_lng=test_flight.longitude(adjusted_time) + np.random.normal(0,GPS_NOISE_STD),
            altimeter_reading=test_flight.z(adjusted_time) + np.random.normal(0,ALT_NOISE_STD),
            gps_height=test_flight.z(adjusted_time) + np.random.normal(0,ALT_NOISE_STD),
            accel_x = test_flight.ax(adjusted_time) + np.random.normal(0,ACC_NOISE_STD),
            accel_y = test_flight.ay(adjusted_time) + np.random.normal(0,ACC_NOISE_STD),
            accel_z = test_flight.az(adjusted_time) + np.random.normal(0,ACC_NOISE_STD),
            v_north = v_enu[1],
            v_east = v_enu[0],
            v_down = -v_enu[2],
            time = time
        )

    def get_orientation(self, time) -> R:
        return R.from_quat(np.array([
            test_flight.e0(time), test_flight.e1(time), test_flight.e2(time), test_flight.e3(time)
        ]), scalar_first=True)
