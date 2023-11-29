from flight_model import test_flight
from utils import TelemetryData
import numpy as np

class Rocket:
    def __init__(self, initial_position):
        self.initial_position = initial_position
        self.initial_alt = test_flight.z(0)

    def get_position(self, time):
        return self.initial_position + np.array([test_flight.x(time), test_flight.y(time), test_flight.z(time)-self.initial_alt])

    def get_telemetry(self, time):
        telem_data = TelemetryData(
            gps_lat=test_flight.latitude(time),
            gps_lng=test_flight.longitude(time),
            altimeter_reading=test_flight.z(time),
        )
        return telem_data