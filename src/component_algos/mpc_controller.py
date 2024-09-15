import line_profiler
from scipy.optimize import least_squares, Bounds
import numpy as np
from src.component_algos.rocket_filter import RocketFilter

class MPCController:
    def __init__(self, time_horizon = 1, n_steps = 10):
        self.time_horizon = time_horizon
        self.n_steps = n_steps
        
    
    @line_profiler.profile
    def step(self, filter: RocketFilter, current_bearing: tuple[float,float]):
        '''
        Returns control input.
        @param current_bearing: (azimuth, altitude) in degrees
        '''
        dt = self.time_horizon/self.n_steps
        filter_copy = filter.copy()
        target_positions = [filter_copy.hx_bearing(filter_copy.x)]
        for _ in range(self.n_steps):
            filter_copy.predict(dt)
            target_positions.append(filter_copy.hx_bearing(filter_copy.x))
        target_positions = np.array(target_positions)
        
        def residuals(u_array: np.ndarray):
            '''
            u is (2*(n_steps+1)) long. The first two are the initial control inputs that we're going to return, and 
            the rest are the control inputs for the n_steps
            '''
            residuals = []
            bearing = np.array(current_bearing)
            for u, position in zip(u_array.reshape(-1,2), target_positions):
                residuals.extend(bearing - position)
                bearing += u * dt
            return np.array(residuals)
        
        bounds_abs = np.tile([8,6], self.n_steps+1)
        u = least_squares(
            fun=residuals, 
            x0=np.zeros(2*(self.n_steps+1)), 
            bounds=Bounds(-bounds_abs, bounds_abs), 
            max_nfev=50
        ).x

        # print(target_positions)
        # print(np.mean(np.square(residuals(u))))
        # print(u.reshape(-1,2))

        return u[:2]
