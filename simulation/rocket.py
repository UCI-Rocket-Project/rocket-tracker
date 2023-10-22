import numpy as np
class Rocket:
    def __init__(self):
        self.height = 0
        self.acceleration = 2*9.81
        self.launch_time = 0
        self.burn_duration = 10
        self.landed = False

    def step(self, time):
        '''
            `time` should be a float, in seconds, since the start of the sim
        '''
        if time<self.launch_time:
            self.height = 0
        elif time<self.launch_time+self.burn_duration:
            # before engine cutoff
            self.height = 0.5*self.acceleration*(time-self.launch_time)**2
        else:
            max_height = 0.5*self.acceleration*(self.burn_duration)**2 
            time_since_cutoff = time-self.burn_duration-self.launch_time
            self.height = max_height - 0.5 * 9.81 * time_since_cutoff**2
        
        if self.height<0:
            self.height=0
            self.landed = True