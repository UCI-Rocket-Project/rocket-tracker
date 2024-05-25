
from datetime import datetime
from zoneinfo import ZoneInfo
from suncalc import get_position
import numpy as np

MOUNT_GPS_LAT = 33.6449307
MOUNT_GPS_LNG = -117.8413249

# get sun position
dates = [
    datetime(2023, 1, 12, 5+i, 0, 0, 0, ZoneInfo("America/Los_Angeles"))
    for i in range(16)
]

for date in dates:
    sun_position = get_position(date, MOUNT_GPS_LAT, MOUNT_GPS_LNG)
    print([np.rad2deg(x) for x in sun_position.values()])