from read_telem import read_telemetry, get_positions
import numpy as np

def eval_cubic(coefficients, t):
    a,b,c,d = coefficients
    return a*t**3 + b*t**2 + c*t + d

def eval_quadratic(coefficients, t):
    a,b,c = coefficients
    return a*t**2+b*t+c

def eval_linear(coefficients, t):
    a,b = coefficients
    return a*t+b
class Rocket:
    def __init__(self):
        self.engine_cutoff_time = 23
        positions = np.array(get_positions(read_telemetry()))
        positions[:,0]-=positions[0,0]

        pre_cutoff = positions[positions[:,0]<self.engine_cutoff_time]
        self.x_coeff_1 = np.polyfit(pre_cutoff[:,0], pre_cutoff[:,1],2)
        self.y_coeff_1 = np.polyfit(pre_cutoff[:,0], pre_cutoff[:,2],2)
        self.z_coeff_1 = np.polyfit(pre_cutoff[:,0], pre_cutoff[:,3],3)
        post_cutoff = positions[positions[:,0]>=self.engine_cutoff_time]
        self.x_coeff_2 = np.polyfit(post_cutoff[:,0], post_cutoff[:,1],1)
        self.y_coeff_2 = np.polyfit(post_cutoff[:,0], post_cutoff[:,2],1)
        self.z_coeff_2 = np.polyfit(post_cutoff[:,0], post_cutoff[:,3],1)
        self.index = 0
        self.position = [0,0,0]
    

    def step(self, time):
        ''')
            `time` should be a float, in seconds, since the start of the sim
        '''
        if time<self.engine_cutoff_time:
            self.position = [
                0,#eval_quadratic(self.x_coeff_1, time),
                0,#eval_quadratic(self.y_coeff_1, time),
                eval_cubic(self.z_coeff_1, time)
            ]
        else:
            self.position = [
                0,#eval_linear(self.x_coeff_2,time),
                0,#eval_linear(self.y_coeff_2,time),
                eval_linear(self.z_coeff_2,time)
            ]