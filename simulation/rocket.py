from flight_model import test_flight
from utils import TelemetryData
import numpy as np

class Rocket:
    def __init__(self, initial_position):
        self.initial_position = initial_position
        self.initial_alt = test_flight.z(0)

    def get_position(self, time):
        return self.initial_position + np.array([test_flight.x(time), test_flight.y(time), test_flight.z(time)-self.initial_alt])

    def get_velocity(self, time):
        return np.array([test_flight.vx(time), test_flight.vy(time), test_flight.vz(time)])
    
    def get_acceleration(self, time):
        return np.array([test_flight.ax(time), test_flight.ay(time), test_flight.az(time)])

    def get_telemetry(self, time):
        telem_data = TelemetryData(
            gps_lat=test_flight.latitude(time),
            gps_lng=test_flight.longitude(time),
            altimeter_reading=test_flight.z(time),
            accel_x = test_flight.ax(time),
            accel_y = test_flight.ay(time),
            accel_z = test_flight.az(time),
        )
        return telem_data