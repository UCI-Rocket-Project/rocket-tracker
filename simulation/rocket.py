from read_telem import read_telemetry, get_positions
import numpy as np

def eval_poly(coefficients, t):
    n = len(coefficients)
    return sum([coefficients[i]*t**(n-i-1) for i in range(n)])

class Rocket:
    def __init__(self, initial_position):
        self.initial_position = initial_position
        self.engine_cutoff_time = 23
        positions = np.array(get_positions(read_telemetry()))
        positions[:,0]-=positions[0,0]

        pre_cutoff = positions[positions[:,0]<self.engine_cutoff_time]
        self.x_coeff_1 = np.polyfit(pre_cutoff[:,0], pre_cutoff[:,1],2)
        self.y_coeff_1 = np.polyfit(pre_cutoff[:,0], pre_cutoff[:,2],2)
        self.z_coeff_1 = np.polyfit(pre_cutoff[:,0], pre_cutoff[:,3],5)
        self.z_coeff_1[-1]=0 # start at h=0
        post_cutoff = positions[positions[:,0]>=self.engine_cutoff_time]
        self.x_coeff_2 = np.polyfit(post_cutoff[:,0], post_cutoff[:,1],1)
        self.y_coeff_2 = np.polyfit(post_cutoff[:,0], post_cutoff[:,2],1)
        self.z_coeff_2 = np.polyfit(post_cutoff[:,0], post_cutoff[:,3],1)
        self.index = 0
        self.position = initial_position 

        self.pad_time = 1
    

    def step(self, time):
        ''')
            `time` should be a float, in seconds, since the start of the sim
        '''
        if time < self.pad_time:
            return
        time -= self.pad_time
        if time<self.engine_cutoff_time:
            self.position = self.initial_position + np.array([
                0,#eval_quadratic(self.x_coeff_1, time),
                0,#eval_quadratic(self.y_coeff_1, time),
                eval_poly(self.z_coeff_1, time)
            ])
        else:
            self.position = self.initial_position + np.array([
                0,#eval_linear(self.x_coeff_2,time),
                0,#eval_linear(self.y_coeff_2,time),
                eval_poly(self.z_coeff_2,time)
            ])