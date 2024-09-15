import line_profiler
from scipy.optimize import lsq_linear, Bounds
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

        dt_matrix = np.repeat(np.tril(dt*np.ones(([self.n_steps+1, 2*(self.n_steps+1)]))), 2, axis=0)
        
        bounds_abs = np.tile([8,6], self.n_steps+1)
        u = lsq_linear(
            dt_matrix,
            target_positions.flatten() - np.tile(np.array(current_bearing), self.n_steps+1),
            bounds=Bounds(-bounds_abs, bounds_abs), 
            # max_nfev=50
        ).x

        # print(target_positions)
        # print(np.mean(np.square(residuals(u))))
        # print(u.reshape(-1,2))

        return u[:2]
