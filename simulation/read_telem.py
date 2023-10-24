from math import sqrt, radians
import matplotlib.pyplot as plt
import os

# scuffed hack to force reading files from this path, even in the debugger
CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def read_telemetry():
    # [timestamp, GPS [latitude, longitude], GYRO [x, y, z], ACC [x, y, z], BMP Altitude]
    telem = []

    with open(f"{CURRENT_FILE_PATH}/telem.csv", "r") as f:

        while True:
            line = f.readline().strip()

            if line == "":
                break

            data = [0, [0, 0], [0, 0, 0], [0, 0, 0], 0]
            if not "GPS" in line:
                continue
            line = line.split(";")

            for section in line:
                if "GPS" in section:
                    section = section.split(",")
                    data[0] = float(section[0][4:])
                    if (section[1] == "A"):
                        if section[3] == "S":
                            data[1][0] = float("-" + section[2][:2] + "." + str(float(section[2][2:]) / 60)[2:8])
                        else:
                            data[1][0] = float(section[2][:2] + "." + str(float(section[2][2:]) / 60)[2:8])

                        if section[5] == "W":
                            data[1][1] = float("-" + section[4][:3] + "." + str(float(section[4][3:]) / 60)[2:8])
                        else:
                            data[1][1] = float(section[4][:3] + "." + str(float(section[4][3:]) / 60)[2:8])

                if "GYRO" in section:
                    section = section.split(",")
                    data[2][0] = float(section[0][5:])
                    data[2][1] = float(section[1])
                    data[2][2] = float(section[2])

                if "ACC" in section:
                    section = section.split(",")
                    data[3][0] = float(section[0][4:])
                    data[3][1] = float(section[1])
                    data[3][2] = float(section[2])

                if "BMP" in section:
                    section = section.split(",")
                    data[4] = float(section[2])

            telem.append(data)
    return telem

def get_positions(telemetry_data):
    '''
    Return ENU coordinates centered on the first telemetry datapoint
    '''
    EARTH_RADIUS = 6_378_000
    initial_lat, initial_lng = telemetry_data[0][1]
    initial_alt = telemetry_data[0][-1]
    initial_timestamp = telemetry_data[0][0]
    positions = []
    for timestamp, (lat,lng), (gyro_x,gyro_y,gyro_z), (acc_x,acc_y,acc_z), alt in telemetry_data:
        if abs(alt-initial_alt)<1:
            continue

        d_lat =  radians(lat-initial_lat)
        d_lng = radians(lng-initial_lng)
        if abs(d_lat)>1 or abs(d_lng)>1:
            continue
        d_alt = alt-initial_alt
        d_time = timestamp-initial_timestamp
        x_est = d_lng*EARTH_RADIUS
        y_est = d_lat*EARTH_RADIUS
        positions.append([d_time, x_est, y_est, d_alt])
    return positions

if __name__=="__main__":
    telem = read_telemetry()
    positions = get_positions(telem)
    times =[p[0] for p in positions]
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]
    hs = [p[3] for p in positions]
    plt.figure()
    plt.title("X drift vs time")
    plt.plot(times, xs)

    plt.figure()
    plt.title("Y drift vs time")
    plt.plot(times, ys)

    plt.figure()
    plt.title("Altitude vs time")
    plt.plot(times, hs)

    plt.figure()
    plt.title("Altitude vs total horizontal drift")
    plt.plot(hs, [sqrt(x**2+y**2) for x,y in zip(xs,ys)])