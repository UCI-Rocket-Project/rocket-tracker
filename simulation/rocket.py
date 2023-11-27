from read_telem import read_telemetry, get_positions
import numpy as np
from utils import TelemetryData

def eval_poly(coefficients, t):
    n = len(coefficients)
    return sum([coefficients[i]*t**(n-i-1) for i in range(n)])

class Rocket:
    def __init__(self, initial_position):
        self.initial_position = initial_position
        self.engine_cutoff_time = 15 
        self.telem_list = read_telemetry()
        positions = np.array(get_positions(self.telem_list))

        # offset timestamps to start at 0
        self.telem_list = list(filter(lambda x: x[0]>=231730, self.telem_list)) # the rocket only starts moving at roughly second 30
        
        initial_timestamp = self.telem_list[0][0]
        for i in range(len(self.telem_list)):
            self.telem_list[i][0] -= initial_timestamp
        positions[:,0]-=positions[0,0]

        pre_cutoff = positions[positions[:,0]<self.engine_cutoff_time]
        self.x_coeff_1 = np.polyfit(pre_cutoff[:,0], pre_cutoff[:,1],1)
        self.y_coeff_1 = np.polyfit(pre_cutoff[:,0], pre_cutoff[:,2],1)
        self.z_coeff_1 = np.polyfit(pre_cutoff[:,0], pre_cutoff[:,3],3)

        # make initial positions match
        self.x_coeff_1[-1] = initial_position[0]
        self.y_coeff_1[-1] = initial_position[1]
        self.z_coeff_1[-1] = initial_position[2]

        post_cutoff = positions[positions[:,0]>=self.engine_cutoff_time] - np.array([self.engine_cutoff_time,0,0,0])
        self.x_coeff_2 = np.polyfit(post_cutoff[:,0], post_cutoff[:,1],1)
        self.y_coeff_2 = np.polyfit(post_cutoff[:,0], post_cutoff[:,2],1)
        self.z_coeff_2 = np.polyfit(post_cutoff[:,0], post_cutoff[:,3],2)
        
        # remove discontinuity
        self.x_coeff_2[-1] = eval_poly(self.x_coeff_1, self.engine_cutoff_time)
        self.y_coeff_2[-1] = eval_poly(self.y_coeff_1, self.engine_cutoff_time)
        self.z_coeff_2[-1] = eval_poly(self.z_coeff_1, self.engine_cutoff_time)

        self.index = 0
        self.position = initial_position 

        self.pad_time = 1
        self.sim_start_time = None
    
    def get_telemetry(self, time) -> TelemetryData:
        '''
        Returns the telemetry data at the given time
        '''
        timestamps = np.array([t[0] for t in self.telem_list])
        timestamp, (lat,lng), gyro, accel, altimeter_reading = self.telem_list[np.argmin(abs(timestamps - time))]
        return TelemetryData(
            gps_lat=lat,
            gps_lng=lng,
            altimeter_reading=altimeter_reading
        )

    def get_position(self, time):
        '''
            `time` should be a float, in seconds, since the start of the sim
        '''
        # x was scaled by 40 to make it be going more straight up. Before the scaling, it goes 40 meters horizontal in the first second
        # but only 15 meters vertical, which is kinda weird. It might be my polyfit code that's wrong or the data is just weird.
        if time<self.engine_cutoff_time:
            self.position = np.array([
                eval_poly(self.x_coeff_1, time)/40,
                eval_poly(self.y_coeff_1, time),
                eval_poly(self.z_coeff_1, time)
            ])
        else:
            self.position = np.array([
                eval_poly(self.x_coeff_2,time-self.engine_cutoff_time)/40,
                eval_poly(self.y_coeff_2,time-self.engine_cutoff_time),
                eval_poly(self.z_coeff_2,time-self.engine_cutoff_time)
            ])
        return self.position

if __name__ == "__main__":
    # plot rocket trajectory
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title("X Position")
    times = np.linspace(0,30,1000)
    rocket = Rocket(np.array([0,0,0]))
    positions = []
    for t in times:
        rocket.step(t)
        positions.append(rocket.position)
    positions = np.array(positions)
    plt.plot(times,positions[:,0])
    plt.figure()
    plt.title("Y Position")
    plt.plot(times,positions[:,1])
    plt.figure()
    plt.title("Z Position")
    plt.plot(times,positions[:,2])
    plt.show()

